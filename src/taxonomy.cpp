#include "dada.h"
#include <Rcpp.h>
#include <RcppParallel.h>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdio>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define NBOOT 100
#define TAX_K 8
#define TAX_N_KMERS (1 << (2 * TAX_K))
#define TAX_REF_VERSION 1
// Max contrib matrix size per thread (~32 MB of floats)
#define CONTRIB_MAX_ELEMS 8388608

using namespace Rcpp;

#ifdef __x86_64
#include <cpuid.h>
#include <emmintrin.h>
#include <immintrin.h>

static bool tax_cpu_has_avx2() {
  static int cached = -1;
  if(cached >= 0) return cached != 0;
  unsigned int eax, ebx, ecx, edx;
  if(__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    cached = (ebx & (1u << 5)) ? 1 : 0;
  } else {
    cached = 0;
  }
  return cached != 0;
}

static inline void add_row_avx2(float *scores, const float *row, size_t ngenus) {
  size_t g = 0;
  for(; g + 8 <= ngenus; g += 8) {
    __m256 s = _mm256_loadu_ps(&scores[g]);
    __m256 r = _mm256_loadu_ps(&row[g]);
    _mm256_storeu_ps(&scores[g], _mm256_add_ps(s, r));
  }
  for(; g + 4 <= ngenus; g += 4) {
    __m128 s = _mm_loadu_ps(&scores[g]);
    __m128 r = _mm_loadu_ps(&row[g]);
    _mm_storeu_ps(&scores[g], _mm_add_ps(s, r));
  }
  for(; g < ngenus; g++) scores[g] += row[g];
}
#endif

static inline void add_row_sse2(float *scores, const float *row, size_t ngenus) {
#ifdef __x86_64
  size_t g = 0;
  for(; g + 4 <= ngenus; g += 4) {
    __m128 s = _mm_loadu_ps(&scores[g]);
    __m128 r = _mm_loadu_ps(&row[g]);
    _mm_storeu_ps(&scores[g], _mm_add_ps(s, r));
  }
  for(; g < ngenus; g++) scores[g] += row[g];
#else
  for(size_t g = 0; g < ngenus; g++) scores[g] += row[g];
#endif
}

static inline void add_contrib_row(float *scores, const float *row, size_t ngenus) {
#ifdef __x86_64
  if(tax_cpu_has_avx2()) add_row_avx2(scores, row, ngenus);
  else add_row_sse2(scores, row, ngenus);
#else
  add_row_sse2(scores, row, ngenus);
#endif
}

static inline int tax_nti(char c) {
  if(c == 'A') return 0;
  if(c == 'C') return 1;
  if(c == 'G') return 2;
  if(c == 'T') return 3;
  return -1;
}

int tax_kmer(const char *seq, unsigned int k) {
  unsigned int j, nti;
  int kmer=0;
  for(j=0; j<k; j++) {
    nti = tax_nti(seq[j]);
    if(nti < 0) { kmer = -1; break; }
    kmer = 4*kmer + nti;
  }
  return(kmer);
}

void tax_kvec(const char *seq, unsigned int k, unsigned char *kvec) {
  unsigned int i;
  unsigned int len = strlen(seq);
  size_t klen = len - k + 1;
  int kmer = 0;
  size_t n_kmers = (1 << (2*k));
  memset(kvec, 0, n_kmers * sizeof(unsigned char));
  for(i=0; i<klen; i++) {
    kmer = tax_kmer(&seq[i], k);
    if(kmer>=0 && kmer<n_kmers) kvec[kmer] = 1;
  }
}

unsigned int tax_karray(const char *seq, unsigned int k, int *karray) {
  unsigned int len = strlen(seq);
  if(len < k) return 0;
  size_t klen = len - k + 1;
  unsigned int j = 0;
  int mask = (1 << (2*k)) - 1;
  bool can_roll = false;
  int kmer = 0;
  for(unsigned int i = 0; i < klen; i++) {
    if(!can_roll) {
      kmer = tax_kmer(&seq[i], k);
      can_roll = (kmer >= 0);
    } else {
      int in_nti = tax_nti(seq[i + k - 1]);
      if(in_nti < 0) { can_roll = false; continue; }
      kmer = ((kmer << 2) | in_nti) & mask;
    }
    if(can_roll) karray[j++] = kmer;
  }
  std::sort(karray, karray + j);
  return(j);
}

static inline float score_genus_row(const float *lgk_v, const int *karray, unsigned int arraylen, float max_logp) {
  float logp = 0.0f;
  unsigned int pos = 0;
  for(; pos + 4 <= arraylen; pos += 4) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(&lgk_v[karray[pos + 4]], 0, 1);
#endif
    logp += lgk_v[karray[pos]];
    if(logp < max_logp) return logp;
    logp += lgk_v[karray[pos + 1]];
    if(logp < max_logp) return logp;
    logp += lgk_v[karray[pos + 2]];
    if(logp < max_logp) return logp;
    logp += lgk_v[karray[pos + 3]];
    if(logp < max_logp) return logp;
  }
  for(; pos < arraylen; pos++) {
    logp += lgk_v[karray[pos]];
    if(logp < max_logp) break;
  }
  return logp;
}

static int get_best_genus(int *karray, float *out_logp, unsigned int arraylen, unsigned int n_kmers, unsigned int ngenus, float *lgk_probability, std::mt19937& gen) {
  unsigned int g;
  float *lgk_v;
  int max_g = -1;
  float logp, max_logp = -FLT_MAX;
  double rv;
  unsigned int nmax=0;
  std::uniform_real_distribution<> cunif(0.0, 1.0);
  for(g=0;g<ngenus;g++) {
    lgk_v = &lgk_probability[g*n_kmers];
    logp = score_genus_row(lgk_v, karray, arraylen, max_logp);
    if(max_logp > 0 || logp>max_logp) {
      max_logp = logp; max_g = g; nmax=1;
    } else if (max_logp == logp) {
      nmax++;
      rv = (double) cunif(gen);
      if(rv < 1.0/nmax) max_g = g;
    }
  }
  *out_logp = max_logp;
  return max_g;
}

static void build_contrib(float *contrib, const int *karray, unsigned int arraylen,
                          unsigned int n_kmers, unsigned int ngenus, float *lgk_probability) {
  for(unsigned int pos = 0; pos < arraylen; pos++) {
    int kmer = karray[pos];
    float *out = contrib + ((size_t)pos * ngenus);
    for(unsigned int g = 0; g < ngenus; g++) {
      out[g] = lgk_probability[((size_t)g * n_kmers) + kmer];
    }
  }
}

static int argmax_scores(float *scores, unsigned int ngenus, std::mt19937& gen, float *out_logp) {
  int max_g = -1;
  float max_logp = -FLT_MAX;
  unsigned int nmax = 0;
  std::uniform_real_distribution<> cunif(0.0, 1.0);
  for(unsigned int g = 0; g < ngenus; g++) {
    float logp = scores[g];
    if(max_logp > 0 || logp > max_logp) {
      max_logp = logp; max_g = (int)g; nmax = 1;
    } else if(max_logp == logp) {
      nmax++;
      if((double)cunif(gen) < 1.0/nmax) max_g = (int)g;
    }
  }
  *out_logp = max_logp;
  return max_g;
}

static int score_boot_contrib(float *contrib, int *bootpos, unsigned int bootlen,
                              unsigned int ngenus, float *scores, std::mt19937& gen, float *out_logp) {
  memset(scores, 0, ((size_t)ngenus) * sizeof(float));
  for(unsigned int i = 0; i < bootlen; i++) {
    add_contrib_row(scores, contrib + ((size_t)bootpos[i] * ngenus), ngenus);
  }
  return argmax_scores(scores, ngenus, gen, out_logp);
}

static inline bool use_contrib_matrix(unsigned int arraylen, size_t ngenus) {
  return ((size_t)arraylen * ngenus) <= CONTRIB_MAX_ELEMS;
}

static float *build_lgk_from_refs(std::vector<std::string> refs, std::vector<int> ref_to_genus,
                                    size_t ngenus, unsigned int k, size_t n_kmers, bool verbose) {
  size_t i, g;
  int kmer;
  float *genus_num_plus1 = (float *) calloc(ngenus, sizeof(float));
  if(genus_num_plus1 == NULL) Rcpp::stop("Memory allocation failed.");
  for(i=0;i<refs.size();i++) genus_num_plus1[ref_to_genus[i]]++;
  for(g=0;g<ngenus;g++) genus_num_plus1[g]++;

  float *kmer_prior = (float *) calloc(n_kmers, sizeof(float));
  float *lgk_v;
  float *lgk_probability = (float *) calloc((ngenus * n_kmers), sizeof(float));
  unsigned char *ref_kv = (unsigned char *) malloc(n_kmers * sizeof(unsigned char));
  if(kmer_prior == NULL || lgk_probability == NULL || ref_kv == NULL) {
    Rcpp::stop("Memory allocation failed.");
  }

  size_t nref = refs.size();
  for(i=0;i<nref;i++) {
    tax_kvec(refs[i].c_str(), k, ref_kv);
    g = ref_to_genus[i];
    lgk_v = &lgk_probability[g*n_kmers];
    for(kmer=0;kmer<(int)n_kmers;kmer++) {
      if(ref_kv[kmer]) {
        lgk_v[kmer]++;
        kmer_prior[kmer]++;
      }
    }
  }
  for(kmer=0;kmer<(int)n_kmers;kmer++) {
    kmer_prior[kmer] = (kmer_prior[kmer] + 0.5)/(1.0 + nref);
  }
  for(g=0;g<ngenus;g++) {
    lgk_v = &lgk_probability[g*n_kmers];
    for(kmer=0;kmer<(int)n_kmers;kmer++) {
      lgk_v[kmer] = logf((lgk_v[kmer] + kmer_prior[kmer])/genus_num_plus1[g]);
    }
  }
  if(verbose) Rprintf("Finished processing reference fasta.\n");
  free(genus_num_plus1);
  free(kmer_prior);
  free(ref_kv);
  return lgk_probability;
}

// Holder for mmap'd or malloc'd lgk matrix
class LgkHolder {
public:
  float *ptr;
  size_t n;
#ifndef _WIN32
  void *map;
  size_t map_len;
#endif
  LgkHolder() : ptr(NULL), n(0) {
#ifndef _WIN32
    map = NULL; map_len = 0;
#endif
  }
  ~LgkHolder() { release(); }
  void release() {
    if(ptr == NULL) return;
#ifndef _WIN32
    if(map != NULL) {
      munmap(map, map_len);
      map = NULL; map_len = 0;
      ptr = NULL; n = 0;
      return;
    }
#endif
    free(ptr);
    ptr = NULL; n = 0;
  }
};

static void lgk_xptr_finalizer(SEXP xp) {
  if(R_ExternalPtrAddr(xp) == NULL) return;
  LgkHolder *h = static_cast<LgkHolder *>(R_ExternalPtrAddr(xp));
  delete h;
  R_ClearExternalPtr(xp);
}

struct AssignParallel : public RcppParallel::Worker
{
  std::vector<std::string> seqs;
  std::vector<std::string> rcs;
  float *lgk_probability;
  int *C_genusmat;
  double *C_unifs;
  int *C_rboot;
  int *C_rval;
  std::size_t seq_offset;

  unsigned int k;
  size_t n_kmers;
  size_t ngenus, nlevel;
  unsigned int max_arraylen;
  unsigned int unifs_per_seq;
  size_t max_contrib;
  bool try_rc;
  bool use_contrib;

  AssignParallel(std::vector<std::string> seqs, std::vector<std::string> rcs, float *lgk_probability,
                 int *C_genusmat, double *C_unifs, int *C_rboot, int *C_rval, std::size_t seq_offset,
                 unsigned int k, size_t n_kmers, size_t ngenus, size_t nlevel, unsigned int max_arraylen,
                 unsigned int unifs_per_seq, size_t max_contrib, bool try_rc, bool use_contrib)
    : seqs(seqs), rcs(rcs), lgk_probability(lgk_probability),
      C_genusmat(C_genusmat), C_unifs(C_unifs), C_rboot(C_rboot), C_rval(C_rval), seq_offset(seq_offset),
      k(k), n_kmers(n_kmers), ngenus(ngenus), nlevel(nlevel), max_arraylen(max_arraylen),
      unifs_per_seq(unifs_per_seq), max_contrib(max_contrib), try_rc(try_rc), use_contrib(use_contrib) {}

  void operator()(std::size_t begin, std::size_t end) {
    size_t i, seqlen;
    unsigned int boot, booti, arraylen, arraylen_rc, bootlen;
    int max_g, max_g_rc, boot_g;
    int karray[9999];
    int karray_rc[9999];
    int bootpos[9999/8 + 1];
    int bootkmers[9999/8 + 1];
    double *unifs;
    float logp, logp_rc;
    thread_local std::mt19937 tax_rng(std::random_device{}());
    thread_local float *contrib_buf = NULL;
    thread_local float *scores_buf = NULL;
    thread_local size_t buf_cap = 0;

    if(use_contrib && (buf_cap < max_contrib || contrib_buf == NULL || scores_buf == NULL)) {
      free(contrib_buf);
      free(scores_buf);
      contrib_buf = (float *) malloc(max_contrib * sizeof(float));
      scores_buf = (float *) malloc(ngenus * sizeof(float));
      if(contrib_buf == NULL || scores_buf == NULL) Rcpp::stop("Memory allocation failed.");
      buf_cap = max_contrib;
    } else if(!use_contrib && scores_buf == NULL) {
      scores_buf = (float *) malloc(ngenus * sizeof(float));
      if(scores_buf == NULL) Rcpp::stop("Memory allocation failed.");
    }

    for(std::size_t j=begin;j<end;j++) {
      seqlen = seqs[j].size();
      std::size_t out_j = seq_offset + j;
      if(seqlen < 50) {
        C_rval[out_j] = NA_INTEGER;
        for(i=0;i<nlevel;i++) C_rboot[out_j*nlevel+i] = 0;
      } else {
        arraylen = tax_karray(seqs[j].c_str(), k, karray);
        max_g = get_best_genus(karray, &logp, arraylen, n_kmers, ngenus, lgk_probability, tax_rng);
        if(try_rc) {
          arraylen_rc = tax_karray(rcs[j].c_str(), k, karray_rc);
          if(arraylen != arraylen_rc) Rcpp::stop("Discrepancy between forward and RC arraylen.");
          max_g_rc = get_best_genus(karray_rc, &logp_rc, arraylen_rc, n_kmers, ngenus, lgk_probability, tax_rng);
          if(logp_rc > logp) {
            max_g = max_g_rc;
            memcpy(karray, karray_rc, arraylen * sizeof(int));
          }
        }
        C_rval[out_j] = max_g+1;
        bootlen = arraylen / 8;
        unifs = &C_unifs[j * unifs_per_seq];
        booti = 0;

        if(use_contrib) {
          build_contrib(contrib_buf, karray, arraylen, n_kmers, ngenus, lgk_probability);
          for(boot=0;boot<NBOOT;boot++) {
            for(i=0;i<bootlen;i++,booti++) bootpos[i] = (int)(arraylen * unifs[booti]);
            boot_g = score_boot_contrib(contrib_buf, bootpos, bootlen, ngenus, scores_buf, tax_rng, &logp);
            for(i=0;i<nlevel;i++) {
              if(C_genusmat[boot_g*nlevel+i] == C_genusmat[max_g*nlevel+i]) C_rboot[out_j*nlevel+i]++;
              else break;
            }
          }
        } else {
          for(boot=0;boot<NBOOT;boot++) {
            for(i=0;i<bootlen;i++,booti++) {
              bootkmers[i] = karray[(int)(arraylen * unifs[booti])];
            }
            std::sort(bootkmers, bootkmers + bootlen);
            boot_g = get_best_genus(bootkmers, &logp, bootlen, n_kmers, ngenus, lgk_probability, tax_rng);
            for(i=0;i<nlevel;i++) {
              if(C_genusmat[boot_g*nlevel+i] == C_genusmat[max_g*nlevel+i]) C_rboot[out_j*nlevel+i]++;
              else break;
            }
          }
        }
      }
    }
  }
};

static Rcpp::List tax_assign_queries(std::vector<std::string> seqs, std::vector<std::string> rcs,
                                     float *lgk_probability, Rcpp::IntegerMatrix genusmat, bool try_rc) {
  size_t i, j;
  unsigned int k = TAX_K;
  size_t n_kmers = TAX_N_KMERS;
  size_t nseq = seqs.size();
  if(nseq == 0) Rcpp::stop("No seqs provided to classify.");
  size_t ngenus = genusmat.nrow();
  size_t nlevel = genusmat.ncol();

  unsigned int max_arraylen = 0;
  for(i=0;i<nseq;i++) {
    unsigned int seqlen = seqs[i].size();
    if((seqlen-k+1) > max_arraylen) max_arraylen = seqlen-k+1;
  }
  size_t max_contrib = ((size_t)max_arraylen) * ngenus;
  bool use_contrib = use_contrib_matrix(max_arraylen, ngenus);

  Rcpp::IntegerVector rval(nseq);
  int *C_rval = (int *) malloc(nseq * sizeof(int));
  Rcpp::IntegerMatrix rboot(nseq, nlevel);
  int *C_rboot = (int *) calloc(nseq * nlevel, sizeof(int));
  int *C_genusmat = (int *) malloc(ngenus * nlevel * sizeof(int));
  if(C_rval == NULL || C_rboot == NULL || C_genusmat == NULL) Rcpp::stop("Memory allocation failed.");
  for(i=0;i<ngenus;i++) {
    for(j=0;j<nlevel;j++) C_genusmat[i*nlevel + j] = genusmat(i,j);
  }

  int BLOCK_SIZE = 128;
  int grain_size = (nseq >= 256) ? 4 : 1;
  unsigned int unifs_per_seq = NBOOT * (max_arraylen / 8);

  for(std::size_t block_start = 0; block_start < nseq; block_start += BLOCK_SIZE) {
    std::size_t block_end = block_start + BLOCK_SIZE;
    if(block_end > nseq) block_end = nseq;
    std::size_t block_n = block_end - block_start;

    Rcpp::NumericVector unifs = Rcpp::runif(block_n * unifs_per_seq);
    double *C_unifs = (double *) malloc(unifs.size() * sizeof(double));
    if(C_unifs == NULL) Rcpp::stop("Memory allocation failed.");
    for(i=0;i<unifs.size();i++) C_unifs[i] = unifs(i);

    std::vector<std::string> seqs_block(seqs.begin() + block_start, seqs.begin() + block_end);
    std::vector<std::string> rcs_block;
    if(try_rc) rcs_block = std::vector<std::string>(rcs.begin() + block_start, rcs.begin() + block_end);

    AssignParallel assignParallel(seqs_block, rcs_block, lgk_probability, C_genusmat, C_unifs, C_rboot, C_rval,
                                  block_start, k, n_kmers, ngenus, nlevel, max_arraylen, unifs_per_seq,
                                  max_contrib, try_rc, use_contrib);
    RcppParallel::parallelFor(0, block_n, assignParallel, grain_size);
    free(C_unifs);
    Rcpp::checkUserInterrupt();
  }

  for(i=0;i<nseq;i++) rval(i) = C_rval[i];
  for(i=0;i<nseq;i++) {
    for(j=0;j<nlevel;j++) rboot(i,j) = C_rboot[i*nlevel + j];
  }

  free(C_rboot);
  free(C_rval);
  free(C_genusmat);

  return Rcpp::List::create(_["tax"]=rval, _["boot"]=rboot);
}

//------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List C_assign_taxonomy2(std::vector<std::string> seqs, std::vector<std::string> rcs, std::vector<std::string> refs, std::vector<int> ref_to_genus, Rcpp::IntegerMatrix genusmat, bool try_rc, bool verbose) {
  size_t i;
  size_t ngenus = genusmat.nrow();
  size_t n_kmers = TAX_N_KMERS;

  for(i=0;i<ref_to_genus.size();i++) {
    ref_to_genus[i] = ref_to_genus[i]-1;
    if(ref_to_genus[i]<0 || ref_to_genus[i] >= (int)ngenus) {
      Rcpp::stop("Invalid map from references to genus.");
    }
  }

  float *lgk_probability = build_lgk_from_refs(refs, ref_to_genus, ngenus, TAX_K, n_kmers, verbose);
  Rcpp::List out = tax_assign_queries(seqs, rcs, lgk_probability, genusmat, try_rc);
  free(lgk_probability);
  return out;
}

//------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List C_assign_taxonomy_prepared(std::vector<std::string> seqs, std::vector<std::string> rcs,
                                      SEXP lgk_xp, Rcpp::IntegerMatrix genusmat, bool try_rc) {
  if(R_ExternalPtrAddr(lgk_xp) == NULL) Rcpp::stop("Invalid taxonomy reference pointer.");
  LgkHolder *h = static_cast<LgkHolder *>(R_ExternalPtrAddr(lgk_xp));
  if(h->ptr == NULL) Rcpp::stop("Taxonomy reference matrix is not loaded.");
  return tax_assign_queries(seqs, rcs, h->ptr, genusmat, try_rc);
}

//------------------------------------------------------------------
// [[Rcpp::export]]
void C_build_taxonomy_ref_file(std::vector<std::string> refs, std::vector<int> ref_to_genus,
                               Rcpp::IntegerMatrix genusmat, std::string bin_file, bool verbose) {
  size_t i;
  size_t ngenus = genusmat.nrow();
  size_t n_kmers = TAX_N_KMERS;
  size_t nfloat = ngenus * n_kmers;

  for(i=0;i<ref_to_genus.size();i++) {
    ref_to_genus[i] = ref_to_genus[i]-1;
    if(ref_to_genus[i]<0 || ref_to_genus[i] >= (int)ngenus) {
      Rcpp::stop("Invalid map from references to genus.");
    }
  }

  float *lgk = build_lgk_from_refs(refs, ref_to_genus, ngenus, TAX_K, n_kmers, verbose);
  FILE *f = fopen(bin_file.c_str(), "wb");
  if(f == NULL) Rcpp::stop("Failed to open taxonomy reference file for writing.");
  size_t written = fwrite(lgk, sizeof(float), nfloat, f);
  fclose(f);
  free(lgk);
  if(written != nfloat) Rcpp::stop("Failed to write complete taxonomy reference file.");
}

//------------------------------------------------------------------
// [[Rcpp::export]]
SEXP C_load_taxonomy_ref_file(std::string bin_file, size_t ngenus) {
  size_t n_kmers = TAX_N_KMERS;
  size_t nfloat = ngenus * n_kmers;
  size_t nbytes = nfloat * sizeof(float);

#ifndef _WIN32
  int fd = open(bin_file.c_str(), O_RDONLY);
  if(fd < 0) Rcpp::stop("Failed to open taxonomy reference file.");
  struct stat sb;
  if(fstat(fd, &sb) != 0 || (size_t)sb.st_size != nbytes) {
    close(fd);
    Rcpp::stop("Taxonomy reference file has unexpected size.");
  }
  void *map = mmap(NULL, nbytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if(map == MAP_FAILED) Rcpp::stop("Failed to memory-map taxonomy reference file.");
  LgkHolder *h = new LgkHolder();
  h->ptr = static_cast<float *>(map);
  h->n = nfloat;
  h->map = map;
  h->map_len = nbytes;
  SEXP xp = R_MakeExternalPtr(h, R_NilValue, R_NilValue);
  R_RegisterCFinalizer(xp, lgk_xptr_finalizer);
  return xp;
#else
  float *buf = (float *) malloc(nbytes);
  if(buf == NULL) Rcpp::stop("Memory allocation failed.");
  FILE *f = fopen(bin_file.c_str(), "rb");
  if(f == NULL) { free(buf); Rcpp::stop("Failed to open taxonomy reference file."); }
  size_t nread = fread(buf, sizeof(float), nfloat, f);
  fclose(f);
  if(nread != nfloat) { free(buf); Rcpp::stop("Failed to read complete taxonomy reference file."); }
  LgkHolder *h = new LgkHolder();
  h->ptr = buf;
  h->n = nfloat;
  SEXP xp = R_MakeExternalPtr(h, R_NilValue, R_NilValue);
  R_RegisterCFinalizer(xp, lgk_xptr_finalizer);
  return xp;
#endif
}

//------------------------------------------------------------------
// [[Rcpp::export]]
void C_release_taxonomy_ref(SEXP lgk_xp) {
  lgk_xptr_finalizer(lgk_xp);
}

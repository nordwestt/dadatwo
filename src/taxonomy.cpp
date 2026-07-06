#include "dada.h"
#include <Rcpp.h>
#include <RcppParallel.h>
#include <random>
#include <algorithm>
#include <cstring>
#define NBOOT 100
  
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
  if(tax_cpu_has_avx2()) {
    add_row_avx2(scores, row, ngenus);
  } else {
    add_row_sse2(scores, row, ngenus);
  }
#else
  add_row_sse2(scores, row, ngenus);
#endif
}

// Returns 0-3 for ACGT, -1 otherwise
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
    if(nti < 0) {
      kmer = -1;
      break;
    }
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
    if(kmer>=0 && kmer<n_kmers) {
      kvec[kmer] = 1;
    }
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
      if(in_nti < 0) {
        can_roll = false;
        continue;
      }
      kmer = ((kmer << 2) | in_nti) & mask;
    }
    if(can_roll) {
      karray[j++] = kmer;
    }
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

int get_best_genus(int *karray, float *out_logp, unsigned int arraylen, unsigned int n_kmers, unsigned int ngenus, float *lgk_probability, std::mt19937& gen) {
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
      max_logp = logp;
      max_g = g;
      nmax=1;
    } else if (max_logp == logp) {
      nmax++;
      rv = (double) cunif(gen);
      if(rv < 1.0/nmax) {
        max_g = g;
      }
    }
  }
  *out_logp = max_logp;
  return max_g;
}

// contrib layout: contrib[pos * ngenus + g] = log prob of kmer at karray[pos] for genus g
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
      max_logp = logp;
      max_g = (int)g;
      nmax = 1;
    } else if(max_logp == logp) {
      nmax++;
      if((double)cunif(gen) < 1.0/nmax) {
        max_g = (int)g;
      }
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


struct AssignParallel : public RcppParallel::Worker
{
  std::vector<std::string> seqs;
  std::vector<std::string> rcs;
  float *lgk_probability;
  int *C_genusmat;
  double *C_unifs;
  int *C_rboot;
  int *C_rboot_tax;
  int *C_rval;
  
  unsigned int k;
  size_t n_kmers;
  size_t ngenus, nlevel;
  unsigned int max_arraylen;
  size_t max_contrib;
  bool try_rc;
  
  AssignParallel(std::vector<std::string> seqs, std::vector<std::string> rcs, float *lgk_probability,
                 int *C_genusmat, double *C_unifs, int *C_rboot, int *C_rboot_tax, int *C_rval, 
                 unsigned int k, size_t n_kmers, size_t ngenus, size_t nlevel, unsigned int max_arraylen,
                 size_t max_contrib, bool try_rc)
    : seqs(seqs), rcs(rcs), lgk_probability(lgk_probability), 
      C_genusmat(C_genusmat), C_unifs(C_unifs), C_rboot(C_rboot), C_rboot_tax(C_rboot_tax), C_rval(C_rval), 
      k(k), n_kmers(n_kmers), ngenus(ngenus), nlevel(nlevel), max_arraylen(max_arraylen),
      max_contrib(max_contrib), try_rc(try_rc) {}

  void operator()(std::size_t begin, std::size_t end) {
    size_t i, seqlen;
    unsigned int boot, booti, arraylen, arraylen_rc, bootlen;
    int max_g, max_g_rc, boot_g;
    int karray[9999];
    int karray_rc[9999];
    int bootpos[9999/8 + 1];
    double *unifs;
    float logp, logp_rc;
    thread_local std::mt19937 tax_rng(std::random_device{}());
    thread_local float *contrib_buf = NULL;
    thread_local float *scores_buf = NULL;
    thread_local size_t buf_cap = 0;

    if(buf_cap < max_contrib || contrib_buf == NULL || scores_buf == NULL) {
      free(contrib_buf);
      free(scores_buf);
      contrib_buf = (float *) malloc(max_contrib * sizeof(float));
      scores_buf = (float *) malloc(ngenus * sizeof(float));
      if(contrib_buf == NULL || scores_buf == NULL) {
        Rcpp::stop("Memory allocation failed.");
      }
      buf_cap = max_contrib;
    }

    for(std::size_t j=begin;j<end;j++) {
      seqlen = seqs[j].size();
      if(seqlen < 50) {
        C_rval[j] = NA_INTEGER;
        for(i=0;i<nlevel;i++) {
          C_rboot[j*nlevel+i] = 0;
        }
        for(boot=0;boot<NBOOT;boot++) {
          C_rboot_tax[j*NBOOT + boot] = NA_INTEGER;
        }
      } else {
        arraylen = tax_karray(seqs[j].c_str(), k, karray);
  
        max_g = get_best_genus(karray, &logp, arraylen, n_kmers, ngenus, lgk_probability, tax_rng);
        if(try_rc) {
          arraylen_rc = tax_karray(rcs[j].c_str(), k, karray_rc);
          if(arraylen != arraylen_rc) { Rcpp::stop("Discrepancy between forward and RC arraylen."); }
          max_g_rc = get_best_genus(karray_rc, &logp_rc, arraylen_rc, n_kmers, ngenus, lgk_probability, tax_rng);
          if(logp_rc > logp) {
            max_g = max_g_rc;
            memcpy(karray, karray_rc, arraylen * sizeof(int));
          }
        }
        
        C_rval[j] = max_g+1;

        build_contrib(contrib_buf, karray, arraylen, n_kmers, ngenus, lgk_probability);
        bootlen = arraylen / 8;
        unifs = &C_unifs[j*max_arraylen];
        booti = 0;
        for(boot=0;boot<NBOOT;boot++) {
          for(i=0;i<bootlen;i++,booti++) {
            bootpos[i] = (int)(arraylen * unifs[booti]);
          }
          boot_g = score_boot_contrib(contrib_buf, bootpos, bootlen, ngenus, scores_buf, tax_rng, &logp);
          C_rboot_tax[j*NBOOT+boot] = boot_g+1;
          for(i=0;i<nlevel;i++) {
            if(C_genusmat[boot_g*nlevel+i] == C_genusmat[max_g*nlevel+i]) {
              C_rboot[j*nlevel+i]++;
            } else {
              break;
            }
          }
        }
      }
    }
  }
};

//------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List C_assign_taxonomy2(std::vector<std::string> seqs, std::vector<std::string> rcs, std::vector<std::string> refs, std::vector<int> ref_to_genus, Rcpp::IntegerMatrix genusmat, bool try_rc, bool verbose) {
  size_t i, j, g;
  int kmer;
  unsigned int k=8;
  size_t n_kmers = (1 << (2*k));
  size_t nseq = seqs.size();
  if(nseq == 0) Rcpp::stop("No seqs provided to classify.");
  size_t nref = refs.size();
  if(nref != ref_to_genus.size()) Rcpp::stop("Length mismatch between number of references and map to genus.");
  size_t ngenus = genusmat.nrow();
  size_t nlevel = genusmat.ncol();
  
  for(i=0;i<ref_to_genus.size();i++) {
    ref_to_genus[i] = ref_to_genus[i]-1;
    if(ref_to_genus[i]<0 || ref_to_genus[i] >= ngenus) {
      Rcpp::stop("Invalid map from references to genus.");
    }
  }
  
  float *genus_num_plus1 = (float *) calloc(ngenus, sizeof(float));
  if(genus_num_plus1 == NULL) Rcpp::stop("Memory allocation failed.");  
  for(i=0;i<nref;i++) {
    genus_num_plus1[ref_to_genus[i]]++;
  }
  for(g=0;g<ngenus;g++) {
    genus_num_plus1[g]++;
  }
  
  float *kmer_prior = (float *) calloc(n_kmers, sizeof(float));
  if(kmer_prior == NULL) Rcpp::stop("Memory allocation failed.");
  float *lgk_v;
  float *lgk_probability = (float *) calloc((ngenus * n_kmers), sizeof(float));
  if(lgk_probability == NULL) Rcpp::stop("Memory allocation failed.");
  
  unsigned char *ref_kv = (unsigned char *) malloc(n_kmers * sizeof(unsigned char));
  if(ref_kv == NULL) Rcpp::stop("Memory allocation failed.");
  
  for(i=0;i<nref;i++) {
    tax_kvec(refs[i].c_str(), k, ref_kv);
    g = ref_to_genus[i];
    lgk_v = &lgk_probability[g*n_kmers];
    for(kmer=0;kmer<n_kmers;kmer++) {
      if(ref_kv[kmer]) { 
        lgk_v[kmer]++;
        kmer_prior[kmer]++;
      }
    }
  }
  
  for(kmer=0;kmer<n_kmers;kmer++) {
    kmer_prior[kmer] = (kmer_prior[kmer] + 0.5)/(1.0 + nref);
  }
  
  for(g=0;g<ngenus;g++) {
    lgk_v = &lgk_probability[g*n_kmers];
    for(kmer=0;kmer<n_kmers;kmer++) {
      lgk_v[kmer] = logf((lgk_v[kmer] + kmer_prior[kmer])/genus_num_plus1[g]);
    }
  }
  
  if(verbose) { Rprintf("Finished processing reference fasta."); }
  
  unsigned int max_arraylen = 0;
  unsigned int seqlen;
  for(i=0;i<nseq;i++) {
    seqlen = seqs[i].size();
    if((seqlen-k+1) > max_arraylen) { max_arraylen = seqlen-k+1; }
  }
  size_t max_contrib = ((size_t)max_arraylen) * ngenus;
  
  Rcpp::NumericVector unifs;
  unifs = Rcpp::runif(nseq*NBOOT*(max_arraylen/8));
  double *C_unifs = (double *) malloc(unifs.size() * sizeof(double));
  for(i=0;i<unifs.size();i++) { C_unifs[i] = unifs(i); }
  
  Rcpp::IntegerVector rval(nseq);
  int *C_rval = (int *) malloc(nseq * sizeof(int));
  Rcpp::IntegerMatrix rboot(nseq, nlevel);
  int *C_rboot = (int *) calloc(nseq * nlevel, sizeof(int));
  Rcpp::IntegerMatrix rboot_tax(nseq, NBOOT);
  int *C_rboot_tax = (int *) malloc(nseq * NBOOT * sizeof(int));
  int *C_genusmat = (int *) malloc(ngenus * nlevel * sizeof(int));
  if(C_rval == NULL || C_rboot == NULL || C_rboot_tax == NULL || C_genusmat == NULL) Rcpp::stop("Memory allocation failed.");
  for(i=0;i<ngenus;i++) {
    for(j=0;j<nlevel;j++) {
      C_genusmat[i*nlevel + j] = genusmat(i,j);
    }
  }
  
  AssignParallel assignParallel(seqs, rcs, lgk_probability, C_genusmat, C_unifs, C_rboot, C_rboot_tax, C_rval, k, n_kmers, ngenus, nlevel, max_arraylen, max_contrib, try_rc);
  int INTERRUPT_BLOCK_SIZE=128;
  int grain_size = (nseq >= 256) ? 4 : 1;
  for(i=0;i<nseq;i+=INTERRUPT_BLOCK_SIZE) {
    j = i+INTERRUPT_BLOCK_SIZE;
    if(j > nseq) { j = nseq; }
    RcppParallel::parallelFor(i, j, assignParallel, grain_size);
    Rcpp::checkUserInterrupt();
  }
  
  for(i=0;i<nseq;i++) {
    rval(i) = C_rval[i];
  }
  for(i=0;i<nseq;i++) {
    for(j=0;j<nlevel;j++) {
      rboot(i,j) = C_rboot[i*nlevel + j];
    }
  }
  for(i=0;i<nseq;i++) {
    for(j=0;j<NBOOT;j++) {
      rboot_tax(i,j) = C_rboot_tax[i*NBOOT + j];
    }
  }
  
  free(C_rboot);
  free(C_rboot_tax);
  free(C_unifs);
  free(C_rval);
  free(C_genusmat);
  free(genus_num_plus1);
  free(kmer_prior);
  free(ref_kv);
  free(lgk_probability);

  return(Rcpp::List::create(_["tax"]=rval, _["boot"]=rboot, _["boot_tax"]=rboot_tax));
}

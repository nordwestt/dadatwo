#include "dada.h"

#ifdef __x86_64
#include <cpuid.h>
#include <immintrin.h>

static bool cpu_has_avx2() {
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

static float horizontal_sum_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

float score_genus_avx2(const float *lgk_v, const int *karray, unsigned int arraylen, float max_logp) {
  float logp = 0.0f;
  unsigned int pos = 0;
  for(; pos + 8 <= arraylen; pos += 8) {
    __m256i idx = _mm256_loadu_si256((const __m256i*)&karray[pos]);
    __m256 vals = _mm256_i32gather_ps(lgk_v, idx, 4);
    logp += horizontal_sum_ps(vals);
    if(logp < max_logp) return logp;
  }
  for(; pos < arraylen; pos++) {
    logp += lgk_v[karray[pos]];
    if(logp < max_logp) break;
  }
  return logp;
}
#endif

float score_genus_scalar(const float *lgk_v, const int *karray, unsigned int arraylen, float max_logp) {
  float logp = 0.0f;
  for(unsigned int pos = 0; pos < arraylen; pos++) {
    logp += lgk_v[karray[pos]];
    if(logp < max_logp) break;
  }
  return logp;
}

float score_genus(const float *lgk_v, const int *karray, unsigned int arraylen, float max_logp) {
#ifdef __x86_64
  if(cpu_has_avx2()) {
    return score_genus_avx2(lgk_v, karray, arraylen, max_logp);
  }
#endif
  return score_genus_scalar(lgk_v, karray, arraylen, max_logp);
}


#pragma once

#include <immintrin.h>
#if LEVEL == 1
void gfmul_new_1(__m256i *Out, const __m256i *A256, const __m256i *B256);
#elif LEVEL == 3
void gfmul_new_3(__m256i *Out, const __m256i *A256, const __m256i *B256);
#elif LEVEL == 5
void gfmul_new_5(__m256i *Out, const __m256i *A256, const __m256i *B256);
#endif

void gf2x_mul_base_pclmul(uint64_t      *c,
                          const uint64_t *a,
                           const uint64_t *b);
                           
#ifndef _GF2X_OPT_H_
#define _GF2X_OPT_H_



#include "stdint.h"
#include "stdlib.h"
#include <immintrin.h>

void gfmul_48_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B);
void gfmul_49_pclmul(__m256i *Out, const __m256i *A256, const __m256i *B256);
void gfmul_96_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256);
void gfmul_97_pclmul(__m256i *Out, const __m256i *A256, const __m256i *B256);
void gfmul_160_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B);
void gfmul_161_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256);

void gfmul_48_vpclmul(__m256i *Out, const __m256i *A, const __m256i *B);
void gfmul_49_vpclmul(__m256i *Out, const __m256i *A, const __m256i *B);
// void gfmul_49_vpclmul(__m256i *Out, const __m256i *A256, const __m256i *B256);
void gfmul_96_vpclmul(__m256i *Out, const __m256i *A256, const __m256i *B256);
void gfmul_97_vpclmul(__m256i *Out, const __m256i *A256, const __m256i *B256);
void gfmul_161_vpclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256);
void gfmul_160_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B);

#endif
/**
 * \file gf2x.c
 * \brief AVX2 implementation of multiplication of two polynomials
 */

#include "gf2x_.h"
#include "parameters.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>

#define VEC_N_ARRAY_SIZE_VEC CEIL_DIVIDE(PARAM_N_MULT, 256) /*!< The number of needed vectors to store PARAM_N bits*/
#define WORD 64
#define LAST64 (PARAM_N >> 6)

__m256i a1_times_a2[VEC_N_256_SIZE_64 >> 1];
__m256i o256[VEC_N_ARRAY_SIZE_VEC];
uint64_t *tmp_reduce = (uint64_t *) o256;

/**
 * @brief Compute o(x) = a(x) mod \f$ X^n - 1\f$
 *
 * This function computes the modular reduction of the polynomial a(x)
 *
 * @param[out] o Pointer to the result
 * @param[in] a256 Pointer to the polynomial a(x)
 */
static inline void reduce(__m256i *o, const __m256i *a256) {
    __m256i r256, carry256;
    uint64_t *a = (uint64_t *)a256;
    static const int32_t dec64 = PARAM_N & 0x3f;
    static int32_t d0;
    int32_t i, i2;

    d0 = WORD - dec64;
    for (i = LAST64; i < (PARAM_N >> 5) - 4; i += 4) {
        r256 = _mm256_lddqu_si256((__m256i const *)(&a[i]));
        r256 = _mm256_srli_epi64(r256, dec64);
        carry256 = _mm256_lddqu_si256((__m256i const *)(&a[i + 1]));
        carry256 = _mm256_slli_epi64(carry256, d0);
        r256 ^= carry256;
        i2 = (i - LAST64) >> 2;
        o256[i2] = a256[i2] ^ r256;
    }

    i = i - LAST64;

    for (; i < LAST64 + 1; i++) {
        uint64_t r = a[i + LAST64] >> dec64;
        uint64_t carry = a[i + LAST64 + 1] << d0;
        r ^= carry;
        tmp_reduce[i] = a[i] ^ r;
    }

    tmp_reduce[LAST64] &= BITMASK(PARAM_N, 64);
    memcpy(o, tmp_reduce, VEC_N_SIZE_BYTES);
}

static inline void divide_by_x_plus_one_128(__m256i *out, __m256i *in, int32_t size){
    __m128i *A = (__m128i *) in;
    __m128i *B = (__m128i *) out;

    B[0] = A[0];
    for(int32_t i = 1; i < size; i++) {
        B[i] = _mm_xor_si128(B[i - 1], A[i]);
    }
}

//len = 1: karat_karat_PCLMULQDQ
static inline void gfmul_1(__m256i *Out, const __m256i *A,  const __m256i *B){
    const __m128i *A128 = (const __m128i *)A, *B128 = (const __m128i *)B;
    __m128i *Out128 = (__m128i *)Out;
    __m128i D1[2];
    __m128i D0[2], D2[2];
    __m128i Al = _mm_loadu_si128(A128);
    __m128i Ah = _mm_loadu_si128(A128 + 1);
    __m128i Bl = _mm_loadu_si128(B128);
    __m128i Bh = _mm_loadu_si128(B128 + 1);

    //	Compute Al.Bl=D0
    __m128i DD0 = _mm_clmulepi64_si128(Al, Bl, 0);
    __m128i DD2 = _mm_clmulepi64_si128(Al, Bl, 0x11);
    __m128i AAlpAAh = _mm_xor_si128(Al, _mm_shuffle_epi32(Al, 0x4e));
    __m128i BBlpBBh = _mm_xor_si128(Bl, _mm_shuffle_epi32(Bl, 0x4e)); 
    __m128i DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
    D0[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
    D0[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

    //	Compute Ah.Bh=D2
    DD0 = _mm_clmulepi64_si128(Ah, Bh, 0);
    DD2 = _mm_clmulepi64_si128(Ah, Bh, 0x11);
    AAlpAAh = _mm_xor_si128(Ah, _mm_shuffle_epi32(Ah, 0x4e));
    BBlpBBh = _mm_xor_si128(Bh, _mm_shuffle_epi32(Bh, 0x4e));
    DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
    D2[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
    D2[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

    // Compute AlpAh.BlpBh=D1
    // Initialisation of AlpAh and BlpBh
    __m128i AlpAh = _mm_xor_si128(Al,Ah);
    __m128i BlpBh = _mm_xor_si128(Bl,Bh);
    DD0 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0);
    DD2 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0x11);
    AAlpAAh = _mm_xor_si128(AlpAh, _mm_shuffle_epi32(AlpAh, 0x4e));
    BBlpBBh = _mm_xor_si128(BlpBh, _mm_shuffle_epi32(BlpBh, 0x4e));
    DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
    D1[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
    D1[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

    __m128i middle = _mm_xor_si128(D0[1], D2[0]);
    Out128[0] = D0[0];
    Out128[1] = middle ^ D0[0] ^ D1[0];
    Out128[2] = middle ^ D1[1] ^ D2[1];
    Out128[3] = D2[1];
}

//len = 3: 3-Karatsuba
static inline void gfmul_3(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    __m256i middle;
    static __m256i aa01[1], bb01[1], aa02[1], bb02[1], aa12[1], bb12[1];
    static __m256i D0[2], D1[2], D2[2], D3[2], D4[2], D5[2];
    a0 = A;
    a1 = A + 1;
    a2 = A + 2;
    b0 = B;
    b1 = B + 1;
    b2 = B + 2;
    for (int16_t i = 0; i < 1; i++)
    {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
    }
    gfmul_1(D3, aa01, bb01);
    gfmul_1(D4, aa02, bb02);
    gfmul_1(D5, aa12, bb12);
    gfmul_1(D0, a0, b0);
    gfmul_1(D1, a1, b1);
    gfmul_1(D2, a2, b2);
    for (int16_t i = 0; i < 1; i++)
    {
        int16_t j = i + 1;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 1] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i] ^ D2[j];
        Out[j + 2] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 4] = D5[j] ^ middle;
        Out[j + 4] = D2[j];
    }
}

//len = 5: 5-Karatsuba
static inline void gfmul_5(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *b0, *a1, *b1, *a2, *b2, * a3, * b3, *a4, *b4;
    static __m256i aa01[1], bb01[1], aa02[1], bb02[1], aa03[1], bb03[1],
               aa04[1], bb04[1], aa12[1], bb12[1], aa13[1], bb13[1],
               aa14[1], bb14[1], aa23[1], bb23[1], aa24[1], bb24[1], aa34[1], bb34[1];
    static __m256i D0[2], D1[2], D2[2], D3[2], D4[2],
               D01[2], D02[2], D03[2], D04[2], D12[2],
               D13[2], D14[2], D23[2], D24[2], D34[2];
    a0 = A;
    a1 = a0 + 1;
    a2 = a1 + 1;
    a3 = a2 + 1;
    a4 = a3 + 1;
    b0 = B;
    b1 = b0 + 1;
    b2 = b1 + 1;
    b3 = b2 + 1;
    b4 = b3 + 1;
    for (int32_t i = 0 ; i < 1 ; i++) {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
        aa03[i] = a0[i] ^ a3[i];
        bb03[i] = b0[i] ^ b3[i];
        aa04[i] = a0[i] ^ a4[i];
        bb04[i] = b0[i] ^ b4[i];
        aa12[i] = a1[i] ^ a2[i];
        bb12[i] = b1[i] ^ b2[i];
        aa13[i] = a1[i] ^ a3[i];
        bb13[i] = b1[i] ^ b3[i];
        aa14[i] = a1[i] ^ a4[i];
        bb14[i] = b1[i] ^ b4[i];
        aa23[i] = a2[i] ^ a3[i];
        bb23[i] = b2[i] ^ b3[i];
        aa24[i] = a2[i] ^ a4[i];
        bb24[i] = b2[i] ^ b4[i];
        aa34[i] = a3[i] ^ a4[i];
        bb34[i] = b3[i] ^ b4[i];
    }
    gfmul_1(D01, aa01, bb01);
    gfmul_1(D02, aa02, bb02);
    gfmul_1(D03, aa03, bb03);
    gfmul_1(D04, aa04, bb04);
    gfmul_1(D12, aa12, bb12);
    gfmul_1(D13, aa13, bb13);
    gfmul_1(D14, aa14, bb14);
    gfmul_1(D23, aa23, bb23);
    gfmul_1(D24, aa24, bb24);
    gfmul_1(D34, aa34, bb34);
    gfmul_1(D0, a0, b0);
    gfmul_1(D1, a1, b1);
    gfmul_1(D2, a2, b2);
    gfmul_1(D3, a3, b3);
    gfmul_1(D4, a4, b4);
    for (int16_t i = 0 ; i < 1 ; i++) {
        int16_t j = i + 1;
        Out[i] = D0[i];
        Out[i + 1] = D0[j] ^ D01[i] ^ D0[i] ^ D1[i];
        Out[i + 2] = D1[i] ^ D02[i] ^ D0[i] ^ D2[i] ^ D01[j] ^ D0[j] ^ D1[j];
        Out[i + 3] = D1[j] ^ D03[i] ^ D0[i] ^ D3[i] ^ D12[i] ^ D1[i] ^ D2[i] ^ D02[j] ^ D0[j] ^ D2[j];
        Out[i + 4] = D2[i] ^ D04[i] ^ D0[i] ^ D4[i] ^ D13[i] ^ D1[i] ^ D3[i] ^ D03[j] ^ D0[j] ^ D3[j] ^ D12[j] ^ D1[j] ^ D2[j];
        Out[i + 5] = D2[j] ^ D14[i] ^ D1[i] ^ D4[i] ^ D23[i] ^ D2[i] ^ D3[i] ^ D04[j] ^ D0[j] ^ D4[j] ^ D13[j] ^ D1[j] ^ D3[j];
        Out[i + 6] = D3[i] ^ D24[i] ^ D2[i] ^ D4[i] ^ D14[j] ^ D1[j] ^ D4[j] ^ D23[j] ^ D2[j] ^ D3[j];
        Out[i + 7] = D3[j] ^ D34[i] ^ D3[i] ^ D4[i] ^ D24[j] ^ D2[j] ^ D4[j];
        Out[i + 8] = D4[i] ^ D34[j] ^ D3[j] ^ D4[j];
        Out[i + 9] = D4[j];
    }
}

//len = 6: 2-Karatsuba
static inline void gfmul_6(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[6], D1[6], D2[6], SAA[3], SBB[3];
    gfmul_3(D0, A, B);
    gfmul_3(D2, (A+3), (B+3));
    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 3;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_3(D1, SAA, SBB);
    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 3;
        int32_t is2 = is + 3;
        int32_t is3 = is2 + 3;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 11: 2-Karatsuba
static inline void gfmul_11(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[12], D1[12], D2[10], SAA[6], SBB[6];
    gfmul_6(D0, A, B);
    gfmul_5(D2, (A+6), (B+6));
    for(int32_t i = 0; i < 5; i++) {
        int32_t is = i + 6;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[5]=A[5];        SBB[5]=B[5];    gfmul_6(D1, SAA, SBB);
    for(int32_t i = 0; i < 4; i++) {
        int32_t is = i + 6;
        int32_t is2 = is + 6;
        int32_t is3 = is2 + 6;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 4; i < 6; i++) {
        int32_t is = i + 6;
        int32_t is2 = is + 6;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 12: 2-Karatsuba
static inline void gfmul_12(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[12], D1[12], D2[12], SAA[6], SBB[6];
    gfmul_6(D0, A, B);
    gfmul_6(D2, (A+6), (B+6));
    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 6;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_6(D1, SAA, SBB);
    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 6;
        int32_t is2 = is + 6;
        int32_t is3 = is2 + 6;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 23: 2-Karatsuba
static inline void gfmul_23(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[24], D1[24], D2[22], SAA[12], SBB[12];
    gfmul_12(D0, A, B);
    gfmul_11(D2, (A+12), (B+12));
    for(int32_t i = 0; i < 11; i++) {
        int32_t is = i + 12;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[11]=A[11];        SBB[11]=B[11];    gfmul_12(D1, SAA, SBB);
    for(int32_t i = 0; i < 10; i++) {
        int32_t is = i + 12;
        int32_t is2 = is + 12;
        int32_t is3 = is2 + 12;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 10; i < 12; i++) {
        int32_t is = i + 12;
        int32_t is2 = is + 12;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 24: 2-Karatsuba
static inline void gfmul_24(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[24], D1[24], D2[24], SAA[12], SBB[12];
    gfmul_12(D0, A, B);
    gfmul_12(D2, (A+12), (B+12));
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 12;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_12(D1, SAA, SBB);
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 12;
        int32_t is2 = is + 12;
        int32_t is3 = is2 + 12;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 47: 2-Karatsuba
static inline void gfmul_47(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[48], D1[48], D2[46], SAA[24], SBB[24];
    gfmul_24(D0, A, B);
    gfmul_23(D2, (A+24), (B+24));
    for(int32_t i = 0; i < 23; i++) {
        int32_t is = i + 24;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[23]=A[23];        SBB[23]=B[23];    gfmul_24(D1, SAA, SBB);
    for(int32_t i = 0; i < 22; i++) {
        int32_t is = i + 24;
        int32_t is2 = is + 24;
        int32_t is3 = is2 + 24;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 22; i < 24; i++) {
        int32_t is = i + 24;
        int32_t is2 = is + 24;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 48: 2-Karatsuba
static inline void gfmul_48(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[48], D1[48], D2[48], SAA[24], SBB[24];
    gfmul_24(D0, A, B);
    gfmul_24(D2, (A+24), (B+24));
    for(int32_t i = 0; i < 24; i++) {
        int32_t is = i + 24;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_24(D1, SAA, SBB);
    for(int32_t i = 0; i < 24; i++) {
        int32_t is = i + 24;
        int32_t is2 = is + 24;
        int32_t is3 = is2 + 24;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 141: TC3_128
static inline void gfmul_141(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[94], W1[96], W2[96], W3[96], W4[94];
    static __m256i tmp[96];
    __m128i zero128;
    zero128 = _mm_setzero_si128();
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[47];
    U2 = (const __m256i *)&A256[94];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[47];
    V2 = (const __m256i *)&B256[94];
    for (int32_t i = 0 ; i < 47 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_47(W1, W2, W3);
    const __m128i *U1_128 = ((const __m128i *) U1);
    const __m128i *V1_128 = ((const __m128i *) V1);
    W0[0] = _mm256_set_m128i(U1_128[0],zero128);
    W4[0] = _mm256_set_m128i(V1_128[0],zero128);
    U1_128 = ((const __m128i *) U1) + 1;
    V1_128 = ((const __m128i *) V1) + 1;
    for(int32_t i = 0; i < 46; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_set_m128i(V1_128[i2+1],V1_128[i2]);
        W4[i1] ^= V2[i];
    }
    W0[47] = _mm256_set_m128i(zero128,U1_128[92]) ^ U2[46];
    W4[47] = _mm256_set_m128i(zero128,V1_128[92]) ^ V2[46];
    for (int32_t i = 0 ; i < 47 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[47] = W0[47];
    W2[47] = W4[47];
    for (int32_t i = 0 ; i < 47 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_48(tmp, W3, W2);
    for (int32_t i = 0 ; i < 96; i++) {
        W3[i] = tmp[i];
    }
    gfmul_48(W2, W0, W4);
    gfmul_47(W4, U2, V2);
    gfmul_47(W0, U0, V0);
    for (int32_t i = 0 ; i < 96 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 94 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_128 = ((const __m128i *) W2) + 1;
    const __m128i * U2_128 = ((const __m128i *) W0) + 1;
    for(int32_t i = 0; i < 93; i++) {
        int32_t i2 = i << 1;
        W2[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W2[i] ^= _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    W2[93]=_mm256_set_m128i(U1_128[187],U2_128[186]^U1_128[186]);
    W2[94]=_mm256_set_m128i(U1_128[189],U1_128[188]);
    W2[95]=_mm256_set_m128i(zero128,U1_128[190]);
    U1_128 = ((const __m128i *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ _mm256_set_m128i(U1_128[0],zero128);
    U1_128 = ((const __m128i *) W4) + 1;
    for(int32_t i = 2; i < 94; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^_mm256_set_m128i(U1_128[i2 - 3],U1_128[i2 - 4]);
    }
    tmp[94] = W2[94] ^ W3[94] ^ _mm256_set_m128i(U1_128[185],U1_128[184]);
    tmp[95] = W2[95] ^ W3[95] ^ _mm256_set_m128i(zero128,U1_128[186]);
    divide_by_x_plus_one_128(W2, tmp, 192);
    U1_128 = ((const __m128i *) W3) + 1;
    U2_128 = ((const __m128i *) W1) + 1;
    for(int32_t i = 0; i < 93; i++) {
        int32_t i2 = i << 1;
        tmp[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]) ^ _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    tmp[93]=_mm256_set_m128i(U1_128[187], U2_128[186]^U1_128[186]);
    tmp[94]=_mm256_set_m128i(U1_128[189],U1_128[188]);
    tmp[95]=_mm256_set_m128i(zero128, U1_128[190]);
    divide_by_x_plus_one_128(W3, tmp, 191);
    for (int32_t i = 0 ; i < 94 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[94] = W2[94];
    W1[95] = W2[95];
    for (int32_t i = 0 ; i < 95 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 47; i++)
    {
        int32_t j = i + 47;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 47] = W1[j] ^ W2[i];
        Out[j + 94] = W2[j] ^ W3[i];
        Out[i + 188] = W3[j] ^ W4[i];
        Out[j + 188] = W4[j];
    }
    Out[141] ^= W1[94];
    Out[142] ^= W1[95];
    Out[188] ^= W2[94];
    Out[189] ^= W2[95];
    Out[235] ^= W3[94];
}

/**
 * @brief Multiply two polynomials modulo \f$ X^n - 1\f$.
 *
 * This functions multiplies a dense polynomial <b>a1</b> (of Hamming weight equal to <b>weight</b>)
 * and a dense polynomial <b>a2</b>. The multiplication is done modulo \f$ X^n - 1\f$.
 *
 * @param[out] o Pointer to the result
 * @param[in] a1 Pointer to a polynomial
 * @param[in] a2 Pointer to a polynomial
 */
void vect_mul(__m256i *o, const __m256i *a1, const __m256i *a2) {
    gfmul_141(a1_times_a2, a1, a2);
    reduce(o, a1_times_a2);
    // clear all
    #ifdef __STDC_LIB_EXT1__
        memset_s(a1_times_a2, 0, (VEC_N_SIZE_64 >> 1) * sizeof(__m256i));
    #else
        memset(a1_times_a2, 0, (VEC_N_SIZE_64 >> 1) * sizeof(__m256i));
    #endif
}
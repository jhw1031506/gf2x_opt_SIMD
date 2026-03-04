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

static inline void divide_by_x_plus_one_256(__m256i *in, __m256i *out, int32_t size){
    out[0] = in[0];
    for(int32_t i = 1; i < size; i++) {
        out[i] = _mm256_xor_si256(out[i - 1], in[i]);
    }
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
static inline void gfmul_1(__m256i *Out,  const __m256i *A,  const __m256i *B){
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

//len = 2: 2-Karatsuba
static inline void gfmul_2(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[2], D1[2], D2[2], SAA[1], SBB[1];
    gfmul_1(D0, A, B);
    gfmul_1(D2, (A+1), (B+1));
    for(int32_t i = 0; i < 1; i++) {
        int32_t is = i + 1;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_1(D1, SAA, SBB);
    for(int32_t i = 0; i < 1; i++) {
        int32_t is = i + 1;
        int32_t is2 = is + 1;
        int32_t is3 = is2 + 1;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 3: 3-Karatsuba
static inline void gfmul_3(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    static __m256i middle;
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

//len = 4: 2-Karatsuba
static inline void gfmul_4(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[4], D1[4], D2[4], SAA[2], SBB[2];
    gfmul_2(D0, A, B);
    gfmul_2(D2, (A+2), (B+2));
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 2;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_2(D1, SAA, SBB);
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 2;
        int32_t is2 = is + 2;
        int32_t is3 = is2 + 2;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
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

//len = 8: 2-Karatsuba
static inline void gfmul_8(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[8], D1[8], D2[8], SAA[4], SBB[4];
    gfmul_4(D0, A, B);
    gfmul_4(D2, (A+4), (B+4));
    for(int32_t i = 0; i < 4; i++) {
        int32_t is = i + 4;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_4(D1, SAA, SBB);
    for(int32_t i = 0; i < 4; i++) {
        int32_t is = i + 4;
        int32_t is2 = is + 4;
        int32_t is3 = is2 + 4;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 9: 3-Karatsuba
static inline void gfmul_9(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    static __m256i middle;
    static __m256i aa01[3], bb01[3], aa02[3], bb02[3], aa12[3], bb12[3];
    static __m256i D0[6], D1[6], D2[6], D3[6], D4[6], D5[6];
    a0 = A;
    a1 = A + 3;
    a2 = A + 6;
    b0 = B;
    b1 = B + 3;
    b2 = B + 6;
    for (int16_t i = 0; i < 3; i++)
    {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
    }
    gfmul_3(D3, aa01, bb01);
    gfmul_3(D4, aa02, bb02);
    gfmul_3(D5, aa12, bb12);
    gfmul_3(D0, a0, b0);
    gfmul_3(D1, a1, b1);
    gfmul_3(D2, a2, b2);
    for (int16_t i = 0; i < 3; i++)
    {
        int16_t j = i + 3;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 3] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i] ^ D2[j];
        Out[j + 6] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 12] = D5[j] ^ middle;
        Out[j + 12] = D2[j];
    }
}

//len = 10: 2-Karatsuba
static inline void gfmul_10(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[10], D1[10], D2[10], SAA[5], SBB[5];
    gfmul_5(D0, A, B);
    gfmul_5(D2, (A+5), (B+5));
    for(int32_t i = 0; i < 5; i++) {
        int32_t is = i + 5;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_5(D1, SAA, SBB);
    for(int32_t i = 0; i < 5; i++) {
        int32_t is = i + 5;
        int32_t is2 = is + 5;
        int32_t is3 = is2 + 5;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
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

//len = 25: TC3_128
static inline void gfmul_25(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    static __m256i U0[9], U1[9], U2[8], V0[9], V1[9], V2[8];
    static __m256i W0[18], W1[18], W2[18], W3[19], W4[16];
    static __m256i tmp[19];
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    for(int32_t i = 0; i < 8; i++) {
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 34]));
        V1[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 34]));
        U2[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 68]));
        V2[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 68]));
    }
    U0[8]= (__m256i){A[32], A[33], 0x0ul, 0x0ul};
    V0[8]= (__m256i){B[32], B[33], 0x0ul, 0x0ul};
    U1[8]= (__m256i){A[66], A[67], 0x0ul, 0x0ul};
    V1[8]= (__m256i){B[66], B[67], 0x0ul, 0x0ul};
    for (int32_t i = 0 ; i < 8 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[8] = U0[8] ^ U1[8];
    W2[8] = V0[8] ^ V1[8];
    gfmul_9(W1, W2, W3);
    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};
    W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};
    U1_64 = ((uint64_t *) U1) + 2;
    V1_64 = ((uint64_t *) V1) + 2;
    for(int32_t i = 0; i < 8; i++) {
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));
        W4[i1] ^= V2[i];
    }
    for (int32_t i = 0 ; i < 9 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0 ; i < 9 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_9(tmp, W3, W2);
    for (int32_t i = 0 ; i < 18; i++) {
        W3[i] = tmp[i];
    }
    gfmul_9(W2, W0, W4);
    gfmul_8(W4, U2, V2);
    gfmul_9(W0, U0, V0);
    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 17 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_64 = ((uint64_t *) W2) + 2;
    uint64_t * U2_64 = ((uint64_t *) W0) + 2;
    for(int32_t i = 0; i < 16; i++) {
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    W2[16]=(__m256i){U2_64[64], U2_64[65], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[64]));
    W2[17]=(__m256i){U1_64[68], U1_64[69], 0ul, 0ul};
    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};
    U1_64 = ((uint64_t *) W4) + 2;
    for(int32_t i = 2; i < 16; i++) {
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));
    }
    tmp[16] = W2[16] ^ W3[16] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[56]));
    tmp[17] = W2[17] ^ W3[17] ^ (__m256i){U1_64[60],U1_64[61], 0ul, 0ul};
    divide_by_x_plus_one_128(W2, tmp, 36);
    U1_64 = ((uint64_t *) W3) + 2;
    U2_64 = ((uint64_t *) W1) + 2;
    for(int32_t i = 0; i < 16; i++) {
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    tmp[16]=(__m256i){U2_64[64], U2_64[65], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[64]));
    tmp[17]=(__m256i){U1_64[68], U1_64[69], 0ul, 0ul};
    divide_by_x_plus_one_128(W3, tmp, 35);
    for (int32_t i = 0 ; i < 16 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[16] ^= W2[16];
    W1[17] = W2[17];
    for (int32_t i = 0 ; i < 17 ; i++) {
        W2[i] ^= W3[i];
    }
    for(int32_t i = 0; i < 16; i++) {
        Out[i] = W0[i];
        Out[i + 17] = W2[i];
        Out[i + 34] = W4[i];
    }
    Out[16] = W0[16];
    Out[33] = W2[16];
    Out[34] ^= W2[17];
    U1_64 = ((uint64_t *) &Out[8]) + 2;
    U2_64 = ((uint64_t *) &Out[25]) + 2;
    __m256i aux;
    for(int32_t i = 0; i < 18; i++) {
        int32_t i4 = i << 2;
        aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);
    }
}

//len = 26: TC3_256
static inline void gfmul_26(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
        static __m256i W0[18], W1[19], W2[20], W3[20], W4[16], tmp[20];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[9];
    U2 = (const __m256i *)&A256[18];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[9];
    V2 = (const __m256i *)&B256[18];
    for (int32_t i = 0 ; i < 8 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[8] = U0[8] ^ U1[8];
    W2[8] = V0[8] ^ V1[8];
    gfmul_9(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 9 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    for (int32_t i = 0 ; i < 9 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[9] = W0[9];
    W2[9] = W4[9];
    for (int32_t i = 0 ; i < 9 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_10(tmp, W3, W2);
    for (int32_t i = 0 ; i < 20 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_10(W2, W0, W4);
    gfmul_8(W4, U2, V2);
    gfmul_9(W0, U0, V0);
    for (int32_t i = 0 ; i < 20 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 18 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 17 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[17] = W2[18];
    W2[18] = W2[19];
    for (int32_t i = 0 ; i < 16 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 16 ; i < 19 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[19] = W3[19];
    for (int32_t i = 0 ; i < 16 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 20);
    for (int32_t i = 0 ; i < 17 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[17] = W3[18];
    tmp[18] = W3[19];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 19);
    for (int32_t i = 0 ; i < 16 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 16 ; i < 18 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[18] = W2[18];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 7; i++) {
        int32_t j = i + 9;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 9] = W1[j] ^ W2[i];
        Out[j + 18] = W2[j] ^ W3[i];
        Out[i + 36] = W3[j] ^ W4[i];
        Out[j + 36] = W4[j];
    }
    for (int32_t i = 7; i < 9; i++) {
        int32_t j = i + 9;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 9] = W1[j] ^ W2[i];
        Out[j + 18] = W2[j] ^ W3[i];
        Out[i + 36] = W3[j] ^ W4[i];
    }
    Out[27] ^= W1[18];
    Out[36] ^= W2[18];
}

//len = 27: TC3_128
static inline void gfmul_27(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[18], W1[20], W2[20], W3[20], W4[18];
    static __m256i tmp[20];
    __m128i zero128;
    zero128 = _mm_setzero_si128();
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[9];
    U2 = (const __m256i *)&A256[18];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[9];
    V2 = (const __m256i *)&B256[18];
    for (int32_t i = 0 ; i < 9 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_9(W1, W2, W3);
    const __m128i *U1_128 = ((const __m128i *) U1);
    const __m128i *V1_128 = ((const __m128i *) V1);
    W0[0] = _mm256_set_m128i(U1_128[0],zero128);
    W4[0] = _mm256_set_m128i(V1_128[0],zero128);
    U1_128 = ((const __m128i *) U1) + 1;
    V1_128 = ((const __m128i *) V1) + 1;
    for(int32_t i = 0; i < 8; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_set_m128i(V1_128[i2+1],V1_128[i2]);
        W4[i1] ^= V2[i];
    }
    W0[9] = _mm256_set_m128i(zero128,U1_128[16]) ^ U2[8];
    W4[9] = _mm256_set_m128i(zero128,V1_128[16]) ^ V2[8];
    for (int32_t i = 0 ; i < 9 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[9] = W0[9];
    W2[9] = W4[9];
    for (int32_t i = 0 ; i < 9 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_10(tmp, W3, W2);
    for (int32_t i = 0 ; i < 20; i++) {
        W3[i] = tmp[i];
    }
    gfmul_10(W2, W0, W4);
    gfmul_9(W4, U2, V2);
    gfmul_9(W0, U0, V0);
    for (int32_t i = 0 ; i < 20 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 18 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_128 = ((const __m128i *) W2) + 1;
    const __m128i * U2_128 = ((const __m128i *) W0) + 1;
    for(int32_t i = 0; i < 17; i++) {
        int32_t i2 = i << 1;
        W2[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W2[i] ^= _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    W2[17]=_mm256_set_m128i(U1_128[35],U2_128[34]^U1_128[34]);
    W2[18]=_mm256_set_m128i(U1_128[37],U1_128[36]);
    W2[19]=_mm256_set_m128i(zero128,U1_128[38]);
    U1_128 = ((const __m128i *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ _mm256_set_m128i(U1_128[0],zero128);
    U1_128 = ((const __m128i *) W4) + 1;
    for(int32_t i = 2; i < 18; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^_mm256_set_m128i(U1_128[i2 - 3],U1_128[i2 - 4]);
    }
    tmp[18] = W2[18] ^ W3[18] ^ _mm256_set_m128i(U1_128[33],U1_128[32]);
    tmp[19] = W2[19] ^ W3[19] ^ _mm256_set_m128i(zero128,U1_128[34]);
    divide_by_x_plus_one_128(W2, tmp, 40);
    U1_128 = ((const __m128i *) W3) + 1;
    U2_128 = ((const __m128i *) W1) + 1;
    for(int32_t i = 0; i < 17; i++) {
        int32_t i2 = i << 1;
        tmp[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]) ^ _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    tmp[17]=_mm256_set_m128i(U1_128[35], U2_128[34]^U1_128[34]);
    tmp[18]=_mm256_set_m128i(U1_128[37],U1_128[36]);
    tmp[19]=_mm256_set_m128i(zero128, U1_128[38]);
    divide_by_x_plus_one_128(W3, tmp, 39);
    for (int32_t i = 0 ; i < 18 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[18] = W2[18];
    W1[19] = W2[19];
    for (int32_t i = 0 ; i < 19 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 9; i++)
    {
        int32_t j = i + 9;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 9] = W1[j] ^ W2[i];
        Out[j + 18] = W2[j] ^ W3[i];
        Out[i + 36] = W3[j] ^ W4[i];
        Out[j + 36] = W4[j];
    }
    Out[27] ^= W1[18];
    Out[28] ^= W1[19];
    Out[36] ^= W2[18];
    Out[37] ^= W2[19];
    Out[45] ^= W3[18];
}

//len = 74: TC3_256
static inline void gfmul_74(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[50], W1[51], W2[52], W3[52], W4[48], tmp[52];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[25];
    U2 = (const __m256i *)&A256[50];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[25];
    V2 = (const __m256i *)&B256[50];
    for (int32_t i = 0 ; i < 24 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[24] = U0[24] ^ U1[24];
    W2[24] = V0[24] ^ V1[24];
    gfmul_25(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 25 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    for (int32_t i = 0 ; i < 25 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[25] = W0[25];
    W2[25] = W4[25];
    for (int32_t i = 0 ; i < 25 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_26(tmp, W3, W2);
    for (int32_t i = 0 ; i < 52 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_26(W2, W0, W4);
    gfmul_24(W4, U2, V2);
    gfmul_25(W0, U0, V0);
    for (int32_t i = 0 ; i < 52 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 50 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 49 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[49] = W2[50];
    W2[50] = W2[51];
    for (int32_t i = 0 ; i < 48 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 48 ; i < 51 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[51] = W3[51];
    for (int32_t i = 0 ; i < 48 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 52);
    for (int32_t i = 0 ; i < 49 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[49] = W3[50];
    tmp[50] = W3[51];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 51);
    for (int32_t i = 0 ; i < 48 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 48 ; i < 50 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[50] = W2[50];
    for (int32_t i = 0 ; i < 50 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 23; i++) {
        int32_t j = i + 25;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 25] = W1[j] ^ W2[i];
        Out[j + 50] = W2[j] ^ W3[i];
        Out[i + 100] = W3[j] ^ W4[i];
        Out[j + 100] = W4[j];
    }
    for (int32_t i = 23; i < 25; i++) {
        int32_t j = i + 25;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 25] = W1[j] ^ W2[i];
        Out[j + 50] = W2[j] ^ W3[i];
        Out[i + 100] = W3[j] ^ W4[i];
    }
    Out[75] ^= W1[50];
    Out[100] ^= W2[50];
}

//len = 76: TC3_256
static inline void gfmul_76(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
        static __m256i W0[52], W1[53], W2[54], W3[54], W4[48], tmp[54];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[26];
    U2 = (const __m256i *)&A256[52];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[26];
    V2 = (const __m256i *)&B256[52];
    for (int32_t i = 0 ; i < 24 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[24] = U0[24] ^ U1[24];
    W2[24] = V0[24] ^ V1[24];
    W3[25] = U0[25] ^ U1[25];
    W2[25] = V0[25] ^ V1[25];
    gfmul_26(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 25 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    
    W0[26] = U1[25];
    W4[26] = V1[25];
    for (int32_t i = 0 ; i < 26 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[26] = W0[26];
    W2[26] = W4[26];
    for (int32_t i = 0 ; i < 26 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_27(tmp, W3, W2);
    for (int32_t i = 0 ; i < 54 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_27(W2, W0, W4);
    gfmul_24(W4, U2, V2);
    gfmul_26(W0, U0, V0);
    for (int32_t i = 0 ; i < 54 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 52 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 51 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[51] = W2[52];
    W2[52] = W2[53];
    for (int32_t i = 0 ; i < 48 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 48 ; i < 53 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[53] = W3[53];
    for (int32_t i = 0 ; i < 48 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 54);
    for (int32_t i = 0 ; i < 51 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[51] = W3[52];
    tmp[52] = W3[53];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 53);
    for (int32_t i = 0 ; i < 48 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 48 ; i < 52 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[52] = W2[52];
    for (int32_t i = 0 ; i < 52 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 22; i++) {
        int32_t j = i + 26;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 26] = W1[j] ^ W2[i];
        Out[j + 52] = W2[j] ^ W3[i];
        Out[i + 104] = W3[j] ^ W4[i];
        Out[j + 104] = W4[j];
    }
    for (int32_t i = 22; i < 26; i++) {
        int32_t j = i + 26;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 26] = W1[j] ^ W2[i];
        Out[j + 52] = W2[j] ^ W3[i];
        Out[i + 104] = W3[j] ^ W4[i];
    }
    Out[78] ^= W1[52];
    Out[104] ^= W2[52];
}

//len = 77: TC3_256
static inline void gfmul_77(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
        static __m256i W0[52], W1[53], W2[54], W3[54], W4[50], tmp[54];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[26];
    U2 = (const __m256i *)&A256[52];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[26];
    V2 = (const __m256i *)&B256[52];
    for (int32_t i = 0 ; i < 25 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[25] = U0[25] ^ U1[25];
    W2[25] = V0[25] ^ V1[25];
    gfmul_26(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 26 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    for (int32_t i = 0 ; i < 26 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[26] = W0[26];
    W2[26] = W4[26];
    for (int32_t i = 0 ; i < 26 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_27(tmp, W3, W2);
    for (int32_t i = 0 ; i < 54 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_27(W2, W0, W4);
    gfmul_25(W4, U2, V2);
    gfmul_26(W0, U0, V0);
    for (int32_t i = 0 ; i < 54 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 52 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 51 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[51] = W2[52];
    W2[52] = W2[53];
    for (int32_t i = 0 ; i < 50 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 50 ; i < 53 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[53] = W3[53];
    for (int32_t i = 0 ; i < 50 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 54);
    for (int32_t i = 0 ; i < 51 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[51] = W3[52];
    tmp[52] = W3[53];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 53);
    for (int32_t i = 0 ; i < 50 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 50 ; i < 52 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[52] = W2[52];
    for (int32_t i = 0 ; i < 52 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 24; i++) {
        int32_t j = i + 26;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 26] = W1[j] ^ W2[i];
        Out[j + 52] = W2[j] ^ W3[i];
        Out[i + 104] = W3[j] ^ W4[i];
        Out[j + 104] = W4[j];
    }
    for (int32_t i = 24; i < 26; i++) {
        int32_t j = i + 26;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 26] = W1[j] ^ W2[i];
        Out[j + 52] = W2[j] ^ W3[i];
        Out[i + 104] = W3[j] ^ W4[i];
    }
    Out[78] ^= W1[52];
    Out[104] ^= W2[52];
}

//len = 226: TC3_256
static inline void gfmul_226(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
        static __m256i W0[152], W1[153], W2[154], W3[154], W4[148], tmp[154];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[76];
    U2 = (const __m256i *)&A256[152];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[76];
    V2 = (const __m256i *)&B256[152];
    for (int32_t i = 0 ; i < 74 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[74] = U0[74] ^ U1[74];
    W2[74] = V0[74] ^ V1[74];
    W3[75] = U0[75] ^ U1[75];
    W2[75] = V0[75] ^ V1[75];
    gfmul_76(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 75 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    
    W0[76] = U1[75];
    W4[76] = V1[75];
    for (int32_t i = 0 ; i < 76 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[76] = W0[76];
    W2[76] = W4[76];
    for (int32_t i = 0 ; i < 76 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_77(tmp, W3, W2);
    for (int32_t i = 0 ; i < 154 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_77(W2, W0, W4);
    gfmul_74(W4, U2, V2);
    gfmul_76(W0, U0, V0);
    for (int32_t i = 0 ; i < 154 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 152 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 151 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[151] = W2[152];
    W2[152] = W2[153];
    for (int32_t i = 0 ; i < 148 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 148 ; i < 153 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[153] = W3[153];
    for (int32_t i = 0 ; i < 148 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 154);
    for (int32_t i = 0 ; i < 151 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[151] = W3[152];
    tmp[152] = W3[153];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 153);
    for (int32_t i = 0 ; i < 148 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 148 ; i < 152 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[152] = W2[152];
    for (int32_t i = 0 ; i < 152 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 72; i++) {
        int32_t j = i + 76;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 76] = W1[j] ^ W2[i];
        Out[j + 152] = W2[j] ^ W3[i];
        Out[i + 304] = W3[j] ^ W4[i];
        Out[j + 304] = W4[j];
    }
    for (int32_t i = 72; i < 76; i++) {
        int32_t j = i + 76;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 76] = W1[j] ^ W2[i];
        Out[j + 152] = W2[j] ^ W3[i];
        Out[i + 304] = W3[j] ^ W4[i];
    }
    Out[228] ^= W1[152];
    Out[304] ^= W2[152];
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
    gfmul_226(a1_times_a2, a1, a2);
    reduce(o, a1_times_a2);
    // clear all
    #ifdef __STDC_LIB_EXT1__
        memset_s(a1_times_a2, 0, (VEC_N_SIZE_64 >> 1) * sizeof(__m256i));
    #else
        memset(a1_times_a2, 0, (VEC_N_SIZE_64 >> 1) * sizeof(__m256i));
    #endif
}
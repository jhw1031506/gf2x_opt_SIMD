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

//len = 1: SB_SB_256
static inline void gfmul_1(__m256i *Out,  const __m256i *A,  const __m256i *B){
    __m256i T0, T1, T2, S0, S1, S2;
    __m256i tmp_a, tmp_b;
    __m256i zero = _mm256_setzero_si256();

    //C[0] = Al*Bl, C[1] = Ah*Bh
    tmp_a = *A;
    tmp_b = *B;
    T0 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x00);
    T1 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x01) 
        ^ _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x10);
    T2 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x11);

    T0 = T0 ^ _mm256_slli_si256(T1,8);
    T2 = T2 ^ _mm256_srli_si256(T1,8);

    S0 = _mm256_permute2x128_si256(T0, T2, 0x20);
    S1 = _mm256_permute2x128_si256(T0, T2, 0x31);

    //S2 = Ah*Bl + Al*Bh (128-bit shuffle)
    tmp_a = _mm256_permute4x64_epi64(tmp_a, 0x4e);
    T0 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x00);
    T1 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x01) 
        ^ _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x10);
    T2 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x11);

    T0 = T0 ^ _mm256_slli_si256(T1,8);
    T2 = T2 ^ _mm256_srli_si256(T1,8);

    S2 = _mm256_permute2x128_si256(T2, T0, 0x20) ^ _mm256_permute2x128_si256(T2, T0, 0x31);

    //
    Out[0] = S0 ^ _mm256_blend_epi32(S2,zero,0x0F);
    Out[1] = S1 ^ _mm256_blend_epi32(S2,zero,0xF0);
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

//len = 3: 2-Karatsuba
static inline void gfmul_3(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[4], D1[4], D2[2], SAA[2], SBB[2];
    gfmul_2(D0, A, B);
    gfmul_1(D2, (A+2), (B+2));
    for(int32_t i = 0; i < 1; i++) {
        int32_t is = i + 2;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[1]=A[1];        SBB[1]=B[1];    gfmul_2(D1, SAA, SBB);
    for(int32_t i = 0; i < 0; i++) {
        int32_t is = i + 2;
        int32_t is2 = is + 2;
        int32_t is3 = is2 + 2;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 2;
        int32_t is2 = is + 2;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
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

//len = 6: 3-Karatsuba
static inline void gfmul_6(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    static __m256i middle;
    static __m256i aa01[2], bb01[2], aa02[2], bb02[2], aa12[2], bb12[2];
    static __m256i D0[4], D1[4], D2[4], D3[4], D4[4], D5[4];
    a0 = A;
    a1 = A + 2;
    a2 = A + 4;
    b0 = B;
    b1 = B + 2;
    b2 = B + 4;
    for (int16_t i = 0; i < 2; i++)
    {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
    }
    gfmul_2(D3, aa01, bb01);
    gfmul_2(D4, aa02, bb02);
    gfmul_2(D5, aa12, bb12);
    gfmul_2(D0, a0, b0);
    gfmul_2(D1, a1, b1);
    gfmul_2(D2, a2, b2);
    for (int16_t i = 0; i < 2; i++)
    {
        int16_t j = i + 2;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 2] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i] ^ D2[j];
        Out[j + 4] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 8] = D5[j] ^ middle;
        Out[j + 8] = D2[j];
    }
}

//len = 7: 2-Karatsuba
static inline void gfmul_7(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[8], D1[8], D2[6], SAA[4], SBB[4];
    gfmul_4(D0, A, B);
    gfmul_3(D2, (A+4), (B+4));
    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 4;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[3]=A[3];        SBB[3]=B[3];    gfmul_4(D1, SAA, SBB);
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 4;
        int32_t is2 = is + 4;
        int32_t is3 = is2 + 4;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 2; i < 4; i++) {
        int32_t is = i + 4;
        int32_t is2 = is + 4;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 12: 3-Karatsuba
static inline void gfmul_12(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    static __m256i middle;
    static __m256i aa01[4], bb01[4], aa02[4], bb02[4], aa12[4], bb12[4];
    static __m256i D0[8], D1[8], D2[8], D3[8], D4[8], D5[8];
    a0 = A;
    a1 = A + 4;
    a2 = A + 8;
    b0 = B;
    b1 = B + 4;
    b2 = B + 8;
    for (int16_t i = 0; i < 4; i++)
    {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
    }
    gfmul_4(D3, aa01, bb01);
    gfmul_4(D4, aa02, bb02);
    gfmul_4(D5, aa12, bb12);
    gfmul_4(D0, a0, b0);
    gfmul_4(D1, a1, b1);
    gfmul_4(D2, a2, b2);
    for (int16_t i = 0; i < 4; i++)
    {
        int16_t j = i + 4;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 4] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i] ^ D2[j];
        Out[j + 8] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 16] = D5[j] ^ middle;
        Out[j + 16] = D2[j];
    }
}

//len = 13: 2-Karatsuba
static inline void gfmul_13(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[14], D1[14], D2[12], SAA[7], SBB[7];
    gfmul_7(D0, A, B);
    gfmul_6(D2, (A+7), (B+7));
    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 7;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[6]=A[6];        SBB[6]=B[6];    gfmul_7(D1, SAA, SBB);
    for(int32_t i = 0; i < 5; i++) {
        int32_t is = i + 7;
        int32_t is2 = is + 7;
        int32_t is3 = is2 + 7;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 5; i < 7; i++) {
        int32_t is = i + 7;
        int32_t is2 = is + 7;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 14: 2-Karatsuba
static inline void gfmul_14(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[14], D1[14], D2[14], SAA[7], SBB[7];
    gfmul_7(D0, A, B);
    gfmul_7(D2, (A+7), (B+7));
    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 7;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_7(D1, SAA, SBB);
    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 7;
        int32_t is2 = is + 7;
        int32_t is3 = is2 + 7;
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

//len = 25: 2-Karatsuba
static inline void gfmul_25(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[26], D1[26], D2[24], SAA[13], SBB[13];
    gfmul_13(D0, A, B);
    gfmul_12(D2, (A+13), (B+13));
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 13;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[12]=A[12];        SBB[12]=B[12];    gfmul_13(D1, SAA, SBB);
    for(int32_t i = 0; i < 11; i++) {
        int32_t is = i + 13;
        int32_t is2 = is + 13;
        int32_t is3 = is2 + 13;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 11; i < 13; i++) {
        int32_t is = i + 13;
        int32_t is2 = is + 13;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 26: 2-Karatsuba
static inline void gfmul_26(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[26], D1[26], D2[26], SAA[13], SBB[13];
    gfmul_13(D0, A, B);
    gfmul_13(D2, (A+13), (B+13));
    for(int32_t i = 0; i < 13; i++) {
        int32_t is = i + 13;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_13(D1, SAA, SBB);
    for(int32_t i = 0; i < 13; i++) {
        int32_t is = i + 13;
        int32_t is2 = is + 13;
        int32_t is3 = is2 + 13;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 27: 2-Karatsuba
static inline void gfmul_27(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[28], D1[28], D2[26], SAA[14], SBB[14];
    gfmul_14(D0, A, B);
    gfmul_13(D2, (A+14), (B+14));
    for(int32_t i = 0; i < 13; i++) {
        int32_t is = i + 14;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[13]=A[13];        SBB[13]=B[13];    gfmul_14(D1, SAA, SBB);
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 14;
        int32_t is2 = is + 14;
        int32_t is3 = is2 + 14;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 12; i < 14; i++) {
        int32_t is = i + 14;
        int32_t is2 = is + 14;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
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
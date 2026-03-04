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

static inline void divide_by_x_plus_one_64(__m256i *out, __m256i *in, int32_t size){
    uint64_t *A = (uint64_t*) in;
    uint64_t *B = (uint64_t*) out;

    B[0] = A[0];
    for(int32_t i = 1; i < size; i++) {
        B[i]= B[i - 1] ^ A[i];
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

//len = 16: 2-Karatsuba
static inline void gfmul_16(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[16], D1[16], D2[16], SAA[8], SBB[8];
    gfmul_8(D0, A, B);
    gfmul_8(D2, (A+8), (B+8));
    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 8;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_8(D1, SAA, SBB);
    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 8;
        int32_t is2 = is + 8;
        int32_t is3 = is2 + 8;
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

//len = 47: TC3_64
static inline void gfmul_47(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    static __m256i UV[96];
    static __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[32], W1[32], W2[32], W3[32], W4[32];
    static  __m256i tmp[94];
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    U0 = UV;
    U1 = U0 + 16;
    U2 = U1 + 16;
    V0 = U2 + 16;
    V1 = V0 + 16;
    V2 = V1 + 16;
    for (int32_t i = 0; i < 15; i++){
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4 + 63]));
        V1[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4 + 63]));
        U2[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4 + 126]));
        V2[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4 + 126]));
    }
    U0[15]= (__m256i){A[60], A[61], A[62], 0x0ul};
    V0[15]= (__m256i){B[60], B[61], B[62], 0x0ul};
    U1[15]= (__m256i){A[123], A[124], A[125], 0x0ul};
    V1[15]= (__m256i){B[123], B[124], B[125], 0x0ul};
    U2[15]= (__m256i){A[186], A[187], 0x0ul, 0x0ul};
    V2[15]= (__m256i){B[186], B[187], 0x0ul, 0x0ul};
    for (int32_t i = 0; i < 16; i++){
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_16(W1, W2, W3);
    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *U2_64 = ((uint64_t *) U2);
    uint64_t *V1_64 = ((uint64_t *) V1);
    uint64_t *V2_64 = ((uint64_t *) V2);
    W0[0] = (__m256i){0ul, U1_64[0], U1_64[1] ^ U2_64[0], U1_64[2] ^ U2_64[1]};
    W4[0] = (__m256i){0ul, V1_64[0], V1_64[1] ^ V2_64[0], V1_64[2] ^ V2_64[1]};
    U1_64 = ((uint64_t *) U1) + 3;
    U2_64 = ((uint64_t *) U2) + 2;
    V1_64 = ((uint64_t *) V1) + 3;
    V2_64 = ((uint64_t *) V2) + 2;
    for (int32_t i = 0; i < 15; i++){
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));
        W0[i1] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
        W4[i1] = _mm256_lddqu_si256((__m256i   *)(& V1_64[i4]));
        W4[i1] ^= _mm256_lddqu_si256((__m256i   *)(& V2_64[i4]));
    }
    for (int32_t i = 0; i < 16; i++){
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0; i < 16; i++){
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_16(tmp, W3, W2);
    for (int32_t i = 0; i < 32; i++){
        W3[i] = tmp[i];
    }
    gfmul_16(W2, W0, W4);
    gfmul_16(W4, U2, V2);
    gfmul_16(W0, U0, V0);
    for (int32_t i = 0; i < 32; i++){
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0; i < 32; i++){
        W1[i] ^= W0[i];
    }
    U1_64 = ((uint64_t *) W2) + 1;
    U2_64 = ((uint64_t *) W0) + 1;
    for (int32_t i = 0; i < 31; i++){
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
    }
    W2[31]=(__m256i){U1_64[124]^U2_64[124], U1_64[125]^U2_64[125], U1_64[126]^U2_64[126], 0x0ul};
    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0] ^ (__m256i){0x0ul, 0x0ul, 0x0ul, U1_64[0]};
    U1_64++;
    for (int32_t i = 1; i < 31; i++){
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i   *)(& U1_64[i4 - 4]));
    }
    tmp[31] = W2[31] ^ W3[31] ^ (__m256i){U1_64[120],U1_64[121],U1_64[122], 0x0ul};
    divide_by_x_plus_one_64(tmp, tmp, 128);
    for (int32_t i = 0; i < 32; i++){
        W2[i] = tmp[i];
    }
    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for (int32_t i = 0; i < 31; i++){
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
    }
    tmp[31]=(__m256i){U1_64[124]^U2_64[124],U1_64[125]^U2_64[125],U1_64[126]^U2_64[126],0x0ul};
    divide_by_x_plus_one_64(tmp, tmp, 127);
    for (int32_t i = 0; i < 32; i++){
        W3[i] = tmp[i];
    }
    for (int32_t i = 0; i < 32; i++){
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 0; i < 32; i++){
        W2[i] ^= W3[i];
    }
        memset((__m256i*)tmp, 0, sizeof(__m256i) * 94);
    for (int32_t i = 0; i < 31; i++){
        tmp[i] = W0[i];
        tmp[i + 63] = W4[i];
    }
        tmp[31] = W0[31];
    U1_64 = ((uint64_t *) &tmp[15]) + 3;
    U2_64 = ((uint64_t *) &tmp[31]) + 2;
    V1_64 = ((uint64_t *) &tmp[47]) + 1;
    for(int32_t i = 0; i < 32; i++) {
        int32_t i4 = i << 2;
        __m256i aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W2[i];
        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& V1_64[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& V1_64[i4]), aux);
    }
    for(int32_t i = 0; i < 94; i++) {
        Out[i] = tmp[i];
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
    __m128i * U2_128 = ((__m128i *) W0) + 1;
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
    U2_128 = ((__m128i *) W1) + 1;
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
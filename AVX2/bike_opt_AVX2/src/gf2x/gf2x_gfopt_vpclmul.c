#include <immintrin.h>
#include <stdint.h>
#include "gf2x_gfopt.h"

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

//len = 1: SB_SB_256
static inline void gfmul_1_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
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
static inline void gfmul_2_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[2], D1[2], D2[2], SAA[1], SBB[1];
    gfmul_1_vpclmul(D0, A, B);
    gfmul_1_vpclmul(D2, (A+1), (B+1));
    for(int32_t i = 0; i < 1; i++) {
        int32_t is = i + 1;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_1_vpclmul(D1, SAA, SBB);
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
static inline void gfmul_3_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[4], D1[4], D2[2], SAA[2], SBB[2];
    gfmul_2_vpclmul(D0, A, B);
    gfmul_1_vpclmul(D2, (A+2), (B+2));
    for(int32_t i = 0; i < 1; i++) {
        int32_t is = i + 2;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    SAA[1]=A[1];        SBB[1]=B[1];    
    gfmul_2_vpclmul(D1, SAA, SBB);
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
static inline void gfmul_4_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[4], D1[4], D2[4], SAA[2], SBB[2];
    gfmul_2_vpclmul(D0, A, B);
    gfmul_2_vpclmul(D2, (A+2), (B+2));
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 2;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_2_vpclmul(D1, SAA, SBB);
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

//len = 5: 3-Karatsuba
static inline void gfmul_5_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    __m256i middle;
    static __m256i aa01[2], bb01[2], aa02[2], bb02[2], aa12[2], bb12[2];
    static __m256i D0[4], D1[4], D2[4], D3[4], D4[4], D5[4];
    a0 = A;
    a1 = A + 2;
    a2 = A + 4;
    b0 = B;
    b1 = B + 2;
    b2 = B + 4;
    for (int16_t i = 0; i < 1; i++)
    {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
    }
    aa01[1] = a0[1] ^ a1[1];
    bb01[1] = b0[1] ^ b1[1];
    aa12[1] = a1[1];
    bb12[1] = b1[1];
    aa02[1] = a0[1];
    bb02[1] = b0[1];
    gfmul_2_vpclmul(D3, aa01, bb01);
    gfmul_2_vpclmul(D4, aa02, bb02);
    gfmul_2_vpclmul(D5, aa12, bb12);
    gfmul_2_vpclmul(D0, a0, b0);
    gfmul_2_vpclmul(D1, a1, b1);
    gfmul_1_vpclmul(D2, a2, b2);
    for (int16_t i = 0; i < 0; i++)
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
    for (int16_t i = 0; i < 2; i++)
    {
        int16_t j = i + 2;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 2] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i];
        Out[j + 4] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 8] = D5[j] ^ middle;
    }
}

//len = 6: 3-Karatsuba
static inline void gfmul_6_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    __m256i middle;
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
    gfmul_2_vpclmul(D3, aa01, bb01);
    gfmul_2_vpclmul(D4, aa02, bb02);
    gfmul_2_vpclmul(D5, aa12, bb12);
    gfmul_2_vpclmul(D0, a0, b0);
    gfmul_2_vpclmul(D1, a1, b1);
    gfmul_2_vpclmul(D2, a2, b2);
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
static inline void gfmul_7_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[8], D1[8], D2[6], SAA[4], SBB[4];
    gfmul_4_vpclmul(D0, A, B);
    gfmul_3_vpclmul(D2, (A+4), (B+4));
    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 4;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[3]=A[3];        SBB[3]=B[3];    
    gfmul_4_vpclmul(D1, SAA, SBB);
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

//len = 8: 2-Karatsuba
static inline void gfmul_8_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[8], D1[8], D2[8], SAA[4], SBB[4];
    gfmul_4_vpclmul(D0, A, B);
    gfmul_4_vpclmul(D2, (A+4), (B+4));
    for(int32_t i = 0; i < 4; i++) {
        int32_t is = i + 4;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_4_vpclmul(D1, SAA, SBB);
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

//len = 11: 3-Karatsuba
static inline void gfmul_11_vpclmul(__m256i *Out,   const __m256i *A,  const  __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    __m256i middle;
    static __m256i aa01[4], bb01[4], aa02[4], bb02[4], aa12[4], bb12[4];
    static __m256i D0[8], D1[8], D2[8], D3[8], D4[8], D5[8];
    a0 = A;
    a1 = A + 4;
    a2 = A + 8;
    b0 = B;
    b1 = B + 4;
    b2 = B + 8;
    for (int16_t i = 0; i < 3; i++)
    {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
    }
    aa01[3] = a0[3] ^ a1[3];
    bb01[3] = b0[3] ^ b1[3];
    aa12[3] = a1[3];
    bb12[3] = b1[3];
    aa02[3] = a0[3];
    bb02[3] = b0[3];
    gfmul_4_vpclmul(D3, aa01, bb01);
    gfmul_4_vpclmul(D4, aa02, bb02);
    gfmul_4_vpclmul(D5, aa12, bb12);
    gfmul_4_vpclmul(D0, a0, b0);
    gfmul_4_vpclmul(D1, a1, b1);
    gfmul_3_vpclmul(D2, a2, b2);
    for (int16_t i = 0; i < 2; i++)
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
    for (int16_t i = 2; i < 4; i++)
    {
        int16_t j = i + 4;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 4] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i];
        Out[j + 8] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 16] = D5[j] ^ middle;
    }
}

//len = 12: 3-Karatsuba
static inline void gfmul_12_vpclmul(__m256i *Out,   const __m256i *A,  const  __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    __m256i middle;
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
    gfmul_4_vpclmul(D3, aa01, bb01);
    gfmul_4_vpclmul(D4, aa02, bb02);
    gfmul_4_vpclmul(D5, aa12, bb12);
    gfmul_4_vpclmul(D0, a0, b0);
    gfmul_4_vpclmul(D1, a1, b1);
    gfmul_4_vpclmul(D2, a2, b2);
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
static inline void gfmul_13_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[14], D1[14], D2[12], SAA[7], SBB[7];
    gfmul_7_vpclmul(D0, A, B);
    gfmul_6_vpclmul(D2, (A+7), (B+7));
    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 7;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[6]=A[6];        SBB[6]=B[6];    
    gfmul_7_vpclmul(D1, SAA, SBB);
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
static inline void gfmul_14_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[14], D1[14], D2[14], SAA[7], SBB[7];
    gfmul_7_vpclmul(D0, A, B);
    gfmul_7_vpclmul(D2, (A+7), (B+7));
    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 7;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_7_vpclmul(D1, SAA, SBB);
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

//len = 15: 2-Karatsuba
static inline void gfmul_15_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[16], D1[16], D2[14], SAA[8], SBB[8];
    gfmul_8_vpclmul(D0, A, B);
    gfmul_7_vpclmul(D2, (A+8), (B+8));
    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 8;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[7]=A[7];        SBB[7]=B[7];    
    gfmul_8_vpclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 8;
        int32_t is2 = is + 8;
        int32_t is3 = is2 + 8;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 6; i < 8; i++) {
        int32_t is = i + 8;
        int32_t is2 = is + 8;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 16: 2-Karatsuba
static inline void gfmul_16_vpclmul(__m256i *Out,   const __m256i *A,  const  __m256i *B){
    static __m256i D0[16], D1[16], D2[16], SAA[8], SBB[8];
    gfmul_8_vpclmul(D0, A, B);
    gfmul_8_vpclmul(D2, (A+8), (B+8));
    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 8;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_8_vpclmul(D1, SAA, SBB);
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

//len = 17: 3-Karatsuba
static inline void gfmul_17_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    __m256i middle;
    static __m256i aa01[6], bb01[6], aa02[6], bb02[6], aa12[6], bb12[6];
    static __m256i D0[12], D1[12], D2[12], D3[12], D4[12], D5[12];
    a0 = A;
    a1 = A + 6;
    a2 = A + 12;
    b0 = B;
    b1 = B + 6;
    b2 = B + 12;
    for (int16_t i = 0; i < 5; i++)
    {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
    }
    aa01[5] = a0[5] ^ a1[5];
    bb01[5] = b0[5] ^ b1[5];
    aa12[5] = a1[5];
    bb12[5] = b1[5];
    aa02[5] = a0[5];
    bb02[5] = b0[5];
    gfmul_6_vpclmul(D3, aa01, bb01);
    gfmul_6_vpclmul(D4, aa02, bb02);
    gfmul_6_vpclmul(D5, aa12, bb12);
    gfmul_6_vpclmul(D0, a0, b0);
    gfmul_6_vpclmul(D1, a1, b1);
    gfmul_5_vpclmul(D2, a2, b2);
    for (int16_t i = 0; i < 4; i++)
    {
        int16_t j = i + 6;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 6] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i] ^ D2[j];
        Out[j + 12] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 24] = D5[j] ^ middle;
        Out[j + 24] = D2[j];
    }
    for (int16_t i = 4; i < 6; i++)
    {
        int16_t j = i + 6;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 6] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i];
        Out[j + 12] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 24] = D5[j] ^ middle;
    }
}

//len = 18: 3-Karatsuba
static inline void gfmul_18_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    __m256i middle;
    static __m256i aa01[6], bb01[6], aa02[6], bb02[6], aa12[6], bb12[6];
    static __m256i D0[12], D1[12], D2[12], D3[12], D4[12], D5[12];
    a0 = A;
    a1 = A + 6;
    a2 = A + 12;
    b0 = B;
    b1 = B + 6;
    b2 = B + 12;
    for (int16_t i = 0; i < 6; i++)
    {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
    }
    gfmul_6_vpclmul(D3, aa01, bb01);
    gfmul_6_vpclmul(D4, aa02, bb02);
    gfmul_6_vpclmul(D5, aa12, bb12);
    gfmul_6_vpclmul(D0, a0, b0);
    gfmul_6_vpclmul(D1, a1, b1);
    gfmul_6_vpclmul(D2, a2, b2);
    for (int16_t i = 0; i < 6; i++)
    {
        int16_t j = i + 6;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 6] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i] ^ D2[j];
        Out[j + 12] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 24] = D5[j] ^ middle;
        Out[j + 24] = D2[j];
    }
}

//len = 20: TC3_256
static inline void gfmul_20_vpclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[14], W1[15], W2[16], W3[16], W4[12], tmp[16];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[7];
    U2 = (const __m256i *)&A256[14];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[7];
    V2 = (const __m256i *)&B256[14];
    for (int32_t i = 0 ; i < 6 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[6] = U0[6] ^ U1[6];
    W2[6] = V0[6] ^ V1[6];
    gfmul_7_vpclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 7 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    for (int32_t i = 0 ; i < 7 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[7] = W0[7];
    W2[7] = W4[7];
    for (int32_t i = 0 ; i < 7 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_8_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_8_vpclmul(W2, W0, W4);
    gfmul_6_vpclmul(W4, U2, V2);
    gfmul_7_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 14 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 13 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[13] = W2[14];
    W2[14] = W2[15];
    for (int32_t i = 0 ; i < 12 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 12 ; i < 15 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[15] = W3[15];
    for (int32_t i = 0 ; i < 12 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 16);
    for (int32_t i = 0 ; i < 13 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[13] = W3[14];
    tmp[14] = W3[15];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 15);
    for (int32_t i = 0 ; i < 12 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 12 ; i < 14 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[14] = W2[14];
    for (int32_t i = 0 ; i < 14 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 5; i++) {
        int32_t j = i + 7;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 7] = W1[j] ^ W2[i];
        Out[j + 14] = W2[j] ^ W3[i];
        Out[i + 28] = W3[j] ^ W4[i];
        Out[j + 28] = W4[j];
    }
    for (int32_t i = 5; i < 7; i++) {
        int32_t j = i + 7;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 7] = W1[j] ^ W2[i];
        Out[j + 14] = W2[j] ^ W3[i];
        Out[i + 28] = W3[j] ^ W4[i];
    }
    Out[21] ^= W1[14];
    Out[28] ^= W2[14];
}

//len = 24: 2-Karatsuba
static inline void gfmul_24_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[24], D1[24], D2[24], SAA[12], SBB[12];
    gfmul_12_vpclmul(D0, A, B);
    gfmul_12_vpclmul(D2, (A+12), (B+12));
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 12;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_12_vpclmul(D1, SAA, SBB);
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
static inline void gfmul_25_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[26], D1[26], D2[24], SAA[13], SBB[13];
    gfmul_13_vpclmul(D0, A, B);
    gfmul_12_vpclmul(D2, (A+13), (B+13));
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 13;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[12]=A[12];        SBB[12]=B[12];    gfmul_13_vpclmul(D1, SAA, SBB);
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
static inline void gfmul_26_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[26], D1[26], D2[26], SAA[13], SBB[13];
    gfmul_13_vpclmul(D0, A, B);
    gfmul_13_vpclmul(D2, (A+13), (B+13));
    for(int32_t i = 0; i < 13; i++) {
        int32_t is = i + 13;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_13_vpclmul(D1, SAA, SBB);
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
static inline void gfmul_27_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[28], D1[28], D2[26], SAA[14], SBB[14];
    gfmul_14_vpclmul(D0, A, B);
    gfmul_13_vpclmul(D2, (A+14), (B+14));
    for(int32_t i = 0; i < 13; i++) {
        int32_t is = i + 14;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[13]=A[13];        SBB[13]=B[13];    
    gfmul_14_vpclmul(D1, SAA, SBB);
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

//len = 28: 2-Karatsuba
static inline void gfmul_28_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[28], D1[28], D2[28], SAA[14], SBB[14];
    gfmul_14_vpclmul(D0, A, B);
    gfmul_14_vpclmul(D2, (A+14), (B+14));
    for(int32_t i = 0; i < 14; i++) {
        int32_t is = i + 14;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_14_vpclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 14; i++) {
        int32_t is = i + 14;
        int32_t is2 = is + 14;
        int32_t is3 = is2 + 14;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 31: 2-Karatsuba
static inline void gfmul_31_vpclmul(__m256i *Out,  const __m256i *A,  const  __m256i *B){
    static __m256i D0[32], D1[32], D2[30], SAA[16], SBB[16];
    gfmul_16_vpclmul(D0, A, B);
    gfmul_15_vpclmul(D2, (A+16), (B+16));
    for(int32_t i = 0; i < 15; i++) {
        int32_t is = i + 16;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[15]=A[15];        SBB[15]=B[15];    
    gfmul_16_vpclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 14; i++) {
        int32_t is = i + 16;
        int32_t is2 = is + 16;
        int32_t is3 = is2 + 16;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 14; i < 16; i++) {
        int32_t is = i + 16;
        int32_t is2 = is + 16;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 32: 2-Karatsuba
static inline void gfmul_32_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[32], D1[32], D2[32], SAA[16], SBB[16];
    gfmul_16_vpclmul(D0, A, B);
    gfmul_16_vpclmul(D2, (A+16), (B+16));
    for(int32_t i = 0; i < 16; i++) {
        int32_t is = i + 16;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_16_vpclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 16; i++) {
        int32_t is = i + 16;
        int32_t is2 = is + 16;
        int32_t is3 = is2 + 16;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 33: TC3_128
static inline void gfmul_33_vpclmul(__m256i *Out,  const  __m256i *A256,  const  __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[22], W1[24], W2[24], W3[24], W4[22];
    static __m256i tmp[24];
    __m128i zero128;
    zero128 = _mm_setzero_si128();
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[11];
    U2 = (const __m256i *)&A256[22];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[11];
    V2 = (const __m256i *)&B256[22];
    for (int32_t i = 0 ; i < 11 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_11_vpclmul(W1, W2, W3);
    const __m128i *U1_128 = ((const __m128i *) U1);
    const __m128i *V1_128 = ((const __m128i *) V1);
    W0[0] = _mm256_set_m128i(U1_128[0],zero128);
    W4[0] = _mm256_set_m128i(V1_128[0],zero128);
    U1_128 = ((const __m128i *) U1) + 1;
    V1_128 = ((const __m128i *) V1) + 1;
    for(int32_t i = 0; i < 10; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_set_m128i(V1_128[i2+1],V1_128[i2]);
        W4[i1] ^= V2[i];
    }
    W0[11] = _mm256_set_m128i(zero128,U1_128[20]) ^ U2[10];
    W4[11] = _mm256_set_m128i(zero128,V1_128[20]) ^ V2[10];
    for (int32_t i = 0 ; i < 11 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[11] = W0[11];
    W2[11] = W4[11];
    for (int32_t i = 0 ; i < 11 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_12_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 24; i++) {
        W3[i] = tmp[i];
    }
    gfmul_12_vpclmul(W2, W0, W4);
    gfmul_11_vpclmul(W4, U2, V2);
    gfmul_11_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 24 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 22 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_128 = ((__m128i *) W2) + 1;
    __m128i * U2_128 = ((__m128i *) W0) + 1;
    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        W2[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W2[i] ^= _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    W2[21]=_mm256_set_m128i(U1_128[43],U2_128[42]^U1_128[42]);
    W2[22]=_mm256_set_m128i(U1_128[45],U1_128[44]);
    W2[23]=_mm256_set_m128i(zero128,U1_128[46]);
    U1_128 = ((__m128i *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ _mm256_set_m128i(U1_128[0],zero128);
    U1_128 = ((__m128i *) W4) + 1;
    for(int32_t i = 2; i < 22; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^_mm256_set_m128i(U1_128[i2 - 3],U1_128[i2 - 4]);
    }
    tmp[22] = W2[22] ^ W3[22] ^ _mm256_set_m128i(U1_128[41],U1_128[40]);
    tmp[23] = W2[23] ^ W3[23] ^ _mm256_set_m128i(zero128,U1_128[42]);
    divide_by_x_plus_one_128(W2, tmp, 48);
    U1_128 = ((__m128i *) W3) + 1;
    U2_128 = ((__m128i *) W1) + 1;
    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        tmp[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]) ^ _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    tmp[21]=_mm256_set_m128i(U1_128[43], U2_128[42]^U1_128[42]);
    tmp[22]=_mm256_set_m128i(U1_128[45],U1_128[44]);
    tmp[23]=_mm256_set_m128i(zero128, U1_128[46]);
    divide_by_x_plus_one_128(W3, tmp, 47);
    for (int32_t i = 0 ; i < 22 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[22] = W2[22];
    W1[23] = W2[23];
    for (int32_t i = 0 ; i < 23 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 11; i++)
    {
        int32_t j = i + 11;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 11] = W1[j] ^ W2[i];
        Out[j + 22] = W2[j] ^ W3[i];
        Out[i + 44] = W3[j] ^ W4[i];
        Out[j + 44] = W4[j];
    }
    Out[33] ^= W1[22];
    Out[34] ^= W1[23];
    Out[44] ^= W2[22];
    Out[45] ^= W2[23];
    Out[55] ^= W3[22];
}

//len = 34: TC3_128
static inline void gfmul_34_vpclmul(__m256i *Out,  const  __m256i *A256,  const  __m256i *B256){
    static __m256i U0[12], U1[12], U2[11], V0[12], V1[12], V2[11];
    static __m256i W0[24], W1[24], W2[24], W3[25], W4[22];
    static __m256i tmp[25];
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    for(int32_t i = 0; i < 11; i++) {
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 46]));
        V1[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 46]));
        U2[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 92]));
        V2[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 92]));
    }
    U0[11]= (__m256i){A[44], A[45], 0x0ul, 0x0ul};
    V0[11]= (__m256i){B[44], B[45], 0x0ul, 0x0ul};
    U1[11]= (__m256i){A[90], A[91], 0x0ul, 0x0ul};
    V1[11]= (__m256i){B[90], B[91], 0x0ul, 0x0ul};
    for (int32_t i = 0 ; i < 11 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[11] = U0[11] ^ U1[11];
    W2[11] = V0[11] ^ V1[11];
    gfmul_12_vpclmul(W1, W2, W3);
    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};
    W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};
    U1_64 = ((uint64_t *) U1) + 2;
    V1_64 = ((uint64_t *) V1) + 2;
    for(int32_t i = 0; i < 11; i++) {
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));
        W4[i1] ^= V2[i];
    }
    for (int32_t i = 0 ; i < 12 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0 ; i < 12 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_12_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 24; i++) {
        W3[i] = tmp[i];
    }
    gfmul_12_vpclmul(W2, W0, W4);
    gfmul_11_vpclmul(W4, U2, V2);
    gfmul_12_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 24 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 23 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_64 = ((uint64_t *) W2) + 2;
    uint64_t * U2_64 = ((uint64_t *) W0) + 2;
    for(int32_t i = 0; i < 22; i++) {
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    W2[22]=(__m256i){U2_64[88], U2_64[89], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[88]));
    W2[23]=(__m256i){U1_64[92], U1_64[93], 0ul, 0ul};
    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};
    U1_64 = ((uint64_t *) W4) + 2;
    for(int32_t i = 2; i < 22; i++) {
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));
    }
    tmp[22] = W2[22] ^ W3[22] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[80]));
    tmp[23] = W2[23] ^ W3[23] ^ (__m256i){U1_64[84],U1_64[85], 0ul, 0ul};
    divide_by_x_plus_one_128(W2, tmp, 48);
    U1_64 = ((uint64_t *) W3) + 2;
    U2_64 = ((uint64_t *) W1) + 2;
    for(int32_t i = 0; i < 22; i++) {
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    tmp[22]=(__m256i){U2_64[88], U2_64[89], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[88]));
    tmp[23]=(__m256i){U1_64[92], U1_64[93], 0ul, 0ul};
    divide_by_x_plus_one_128(W3, tmp, 47);
    for (int32_t i = 0 ; i < 22 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[22] ^= W2[22];
    W1[23] = W2[23];
    for (int32_t i = 0 ; i < 23 ; i++) {
        W2[i] ^= W3[i];
    }
    for(int32_t i = 0; i < 22; i++) {
        Out[i] = W0[i];
        Out[i + 23] = W2[i];
        Out[i + 46] = W4[i];
    }
    Out[22] = W0[22];
    Out[45] = W2[22];
    Out[46] ^= W2[23];
    U1_64 = ((uint64_t *) &Out[11]) + 2;
    U2_64 = ((uint64_t *) &Out[34]) + 2;
    __m256i aux;
    for(int32_t i = 0; i < 24; i++) {
        int32_t i4 = i << 2;
        aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);
    }
}

//len = 52: TC3_128
static inline void gfmul_52_vpclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    static __m256i U0[18], U1[18], U2[17], V0[18], V1[18], V2[17];
    static __m256i W0[36], W1[36], W2[36], W3[37], W4[34];
    static __m256i tmp[37];
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    for(int32_t i = 0; i < 17; i++) {
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 70]));
        V1[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 70]));
        U2[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 140]));
        V2[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 140]));
    }
    U0[17]= (__m256i){A[68], A[69], 0x0ul, 0x0ul};
    V0[17]= (__m256i){B[68], B[69], 0x0ul, 0x0ul};
    U1[17]= (__m256i){A[138], A[139], 0x0ul, 0x0ul};
    V1[17]= (__m256i){B[138], B[139], 0x0ul, 0x0ul};
    for (int32_t i = 0 ; i < 17 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[17] = U0[17] ^ U1[17];
    W2[17] = V0[17] ^ V1[17];
    gfmul_18_vpclmul(W1, W2, W3);
    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};
    W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};
    U1_64 = ((uint64_t *) U1) + 2;
    V1_64 = ((uint64_t *) V1) + 2;
    for(int32_t i = 0; i < 17; i++) {
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));
        W4[i1] ^= V2[i];
    }
    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0 ; i < 18 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_18_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 36; i++) {
        W3[i] = tmp[i];
    }
    gfmul_18_vpclmul(W2, W0, W4);
    gfmul_17_vpclmul(W4, U2, V2);
    gfmul_18_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 36 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 35 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_64 = ((uint64_t *) W2) + 2;
    uint64_t * U2_64 = ((uint64_t *) W0) + 2;
    for(int32_t i = 0; i < 34; i++) {
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    W2[34]=(__m256i){U2_64[136], U2_64[137], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[136]));
    W2[35]=(__m256i){U1_64[140], U1_64[141], 0ul, 0ul};
    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};
    U1_64 = ((uint64_t *) W4) + 2;
    for(int32_t i = 2; i < 34; i++) {
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));
    }
    tmp[34] = W2[34] ^ W3[34] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[128]));
    tmp[35] = W2[35] ^ W3[35] ^ (__m256i){U1_64[132],U1_64[133], 0ul, 0ul};
    divide_by_x_plus_one_128(W2, tmp, 72);
    U1_64 = ((uint64_t *) W3) + 2;
    U2_64 = ((uint64_t *) W1) + 2;
    for(int32_t i = 0; i < 34; i++) {
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    tmp[34]=(__m256i){U2_64[136], U2_64[137], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[136]));
    tmp[35]=(__m256i){U1_64[140], U1_64[141], 0ul, 0ul};
    divide_by_x_plus_one_128(W3, tmp, 71);
    for (int32_t i = 0 ; i < 34 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[34] ^= W2[34];
    W1[35] = W2[35];
    for (int32_t i = 0 ; i < 35 ; i++) {
        W2[i] ^= W3[i];
    }
    for(int32_t i = 0; i < 34; i++) {
        Out[i] = W0[i];
        Out[i + 35] = W2[i];
        Out[i + 70] = W4[i];
    }
    Out[34] = W0[34];
    Out[69] = W2[34];
    Out[70] ^= W2[35];
    U1_64 = ((uint64_t *) &Out[17]) + 2;
    U2_64 = ((uint64_t *) &Out[52]) + 2;
    __m256i aux;
    for(int32_t i = 0; i < 36; i++) {
        int32_t i4 = i << 2;
        aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);
    }
}

//len = 53: 2-Karatsuba
static inline void gfmul_53_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[54], D1[54], D2[52], SAA[27], SBB[27];
    gfmul_27_vpclmul(D0, A, B);
    gfmul_26_vpclmul(D2, (A+27), (B+27));
    for(int32_t i = 0; i < 26; i++) {
        int32_t is = i + 27;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[26]=A[26];        SBB[26]=B[26];    
    gfmul_27_vpclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 25; i++) {
        int32_t is = i + 27;
        int32_t is2 = is + 27;
        int32_t is3 = is2 + 27;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 25; i < 27; i++) {
        int32_t is = i + 27;
        int32_t is2 = is + 27;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

// //len = 54: 2-Karatsuba
// static inline void gfmul_54_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[54], D1[54], D2[54], SAA[27], SBB[27];
//     gfmul_27_vpclmul(D0, A, B);
//     gfmul_27_vpclmul(D2, (A+27), (B+27));
//     for(int32_t i = 0; i < 27; i++) {
//         int32_t is = i + 27;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//     gfmul_27_vpclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 27; i++) {
//         int32_t is = i + 27;
//         int32_t is2 = is + 27;
//         int32_t is3 = is2 + 27;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
// }

//len = 54: TC3_256
static inline void gfmul_54_vpclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[36], W1[39], W2[40], W3[40], W4[36], tmp[40];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[18];
    U2 = (const __m256i *)&A256[36];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[18];
    V2 = (const __m256i *)&B256[36];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_18_vpclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 18 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    W0[19] = U2[17];
    W4[19] = V2[17];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[18] = W0[18];
    W3[19] = W0[19];
    W2[18] = W4[18];
    W2[19] = W4[19];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_20_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 40 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_20_vpclmul(W2, W0, W4);
    gfmul_18_vpclmul(W4, U2, V2);
    gfmul_18_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 40 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 36 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 35 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[35] = W2[36];
    W2[36] = W2[37];
    W2[37] = W2[38];
    W2[38] = W2[39];
    for (int32_t i = 0 ; i < 36 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 36 ; i < 39 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[39] = W3[39];
    for (int32_t i = 0 ; i < 36 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 40);
    for (int32_t i = 0 ; i < 35 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[35] = W3[36];
    tmp[36] = W3[37];
    tmp[37] = W3[38];
    tmp[38] = W3[39];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 39);
    for (int32_t i = 0 ; i < 36 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 36 ; i < 39 ; i++) {
        W1[i] = W2[i];
    }
    for (int32_t i = 0 ; i < 38 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 18; i++) {
        int32_t j = i + 18;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 18] = W1[j] ^ W2[i];
        Out[j + 36] = W2[j] ^ W3[i];
        Out[i + 72] = W3[j] ^ W4[i];
        Out[j + 72] = W4[j];
    }
    Out[54] ^= W1[36];
    Out[55] ^= W1[37];
    Out[56] ^= W1[38];
    Out[72] ^= W2[36];
    Out[73] ^= W2[37];
    Out[74] ^= W2[38];
    Out[90] ^= W3[36];
    Out[91] ^= W3[37];
}

//len = 55: 2-Karatsuba
static inline void gfmul_55_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[56], D1[56], D2[54], SAA[28], SBB[28];
    gfmul_28_vpclmul(D0, A, B);
    gfmul_27_vpclmul(D2, (A+28), (B+28));
    for(int32_t i = 0; i < 27; i++) {
        int32_t is = i + 28;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[27]=A[27];        SBB[27]=B[27];    
    gfmul_28_vpclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 26; i++) {
        int32_t is = i + 28;
        int32_t is2 = is + 28;
        int32_t is3 = is2 + 28;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 26; i < 28; i++) {
        int32_t is = i + 28;
        int32_t is2 = is + 28;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 80: TC3_256
static inline void gfmul_80_vpclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[54], W1[55], W2[56], W3[56], W4[52], tmp[56];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[27];
    U2 = (const __m256i *)&A256[54];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[27];
    V2 = (const __m256i *)&B256[54];
    for (int32_t i = 0 ; i < 26 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[26] = U0[26] ^ U1[26];
    W2[26] = V0[26] ^ V1[26];
    gfmul_27_vpclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 27 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    for (int32_t i = 0 ; i < 27 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[27] = W0[27];
    W2[27] = W4[27];
    for (int32_t i = 0 ; i < 27 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_28_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 56 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_28_vpclmul(W2, W0, W4);
    gfmul_26_vpclmul(W4, U2, V2);
    gfmul_27_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 56 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 54 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 53 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[53] = W2[54];
    W2[54] = W2[55];
    for (int32_t i = 0 ; i < 52 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 52 ; i < 55 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[55] = W3[55];
    for (int32_t i = 0 ; i < 52 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 56);
    for (int32_t i = 0 ; i < 53 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[53] = W3[54];
    tmp[54] = W3[55];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 55);
    for (int32_t i = 0 ; i < 52 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 52 ; i < 54 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[54] = W2[54];
    for (int32_t i = 0 ; i < 54 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 25; i++) {
        int32_t j = i + 27;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 27] = W1[j] ^ W2[i];
        Out[j + 54] = W2[j] ^ W3[i];
        Out[i + 108] = W3[j] ^ W4[i];
        Out[j + 108] = W4[j];
    }
    for (int32_t i = 25; i < 27; i++) {
        int32_t j = i + 27;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 27] = W1[j] ^ W2[i];
        Out[j + 54] = W2[j] ^ W3[i];
        Out[i + 108] = W3[j] ^ W4[i];
    }
    Out[81] ^= W1[54];
    Out[108] ^= W2[54];
}

////////////////////////////////

//len = 48: 2-Karatsuba
void gfmul_48_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[48], D1[48], D2[48], SAA[24], SBB[24];
    gfmul_24_vpclmul(D0, A, B);
    gfmul_24_vpclmul(D2, (A+24), (B+24));
    for(int32_t i = 0; i < 24; i++) {
        int32_t is = i + 24;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_24_vpclmul(D1, SAA, SBB);
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

// //len = 49: TC3_256
// void gfmul_49_vpclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
//     const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
//         static __m256i W0[34], W1[35], W2[36], W3[36], W4[30], tmp[36];
//     static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
//     U0 = (const __m256i *)&A256[0];
//     U1 = (const __m256i *)&A256[17];
//     U2 = (const __m256i *)&A256[34];
//     V0 = (const __m256i *)&B256[0];
//     V1 = (const __m256i *)&B256[17];
//     V2 = (const __m256i *)&B256[34];
//     for (int32_t i = 0 ; i < 15 ; i++) {
//         W3[i] = U0[i] ^ U1[i] ^ U2[i];
//         W2[i] = V0[i] ^ V1[i] ^ V2[i];
//     }
//     W3[15] = U0[15] ^ U1[15];
//     W2[15] = V0[15] ^ V1[15];
//     W3[16] = U0[16] ^ U1[16];
//     W2[16] = V0[16] ^ V1[16];
//     gfmul_17_vpclmul(W1, W2, W3);
//     W0[0] = zero;
//     W4[0] = zero;
//     W0[1] = U1[0];
//     W4[1] = V1[0];
//     for (int32_t i = 1 ; i < 16 ; i++) {
//         W0[i + 1] = U1[i] ^ U2[i - 1];
//         W4[i + 1] = V1[i] ^ V2[i - 1];
//     }
//     W0[17] = U1[16];
//     W4[17] = V1[16];
//     for (int32_t i = 0 ; i < 17 ; i++) {
//         W3[i] ^= W0[i];
//         W2[i] ^= W4[i];
//     }
//     W3[17] = W0[17];
//     W2[17] = W4[17];
//     for (int32_t i = 0 ; i < 17 ; i++) {
//         W0[i] ^= U0[i];
//         W4[i] ^= V0[i];
//     }
//     gfmul_18_vpclmul(tmp, W3, W2);
//     for (int32_t i = 0 ; i < 36 ; i++) {
//         W3[i] = tmp[i];
//     }
//     gfmul_18_vpclmul(W2, W0, W4);
//     gfmul_15_vpclmul(W4, U2, V2);
//     gfmul_17_vpclmul(W0, U0, V0);
//     for (int32_t i = 0 ; i < 36 ; i++) {
//         W3[i] ^= W2[i];
//     }
//     for (int32_t i = 0 ; i < 34 ; i++) {
//         W1[i] ^= W0[i];
//     }
//     for (int32_t i = 0 ; i < 33 ; i++) {
//         int32_t i1 = i + 1;
//         W2[i] = W2[i1] ^ W0[i1];
//     }
//     W2[33] = W2[34];
//     W2[34] = W2[35];
//     for (int32_t i = 0 ; i < 30 ; i++) {
//         tmp[i] = W2[i] ^ W3[i] ^ W4[i];
//     }
//     for (int32_t i = 30 ; i < 35 ; i++) {
//         tmp[i] = W2[i] ^ W3[i];
//     }
//     tmp[35] = W3[35];
//     for (int32_t i = 0 ; i < 30 ; i++) {
//         tmp[i + 3] ^= W4[i];
//     }
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 36);
//     for (int32_t i = 0 ; i < 33 ; i++) {
//         int32_t i1 = i + 1;
//         tmp[i] = W3[i1] ^ W1[i1];
//     }
//     tmp[33] = W3[34];
//     tmp[34] = W3[35];
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 35);
//     for (int32_t i = 0 ; i < 30 ; i++) {
//         W1[i] ^= W2[i] ^ W4[i];
//     }
//     for (int32_t i = 30 ; i < 34 ; i++) {
//         W1[i] ^= W2[i];
//     }
//     W1[34] = W2[34];
//     for (int32_t i = 0 ; i < 34 ; i++) {
//         W2[i] ^= W3[i];
//     }
//     for (int32_t i = 0; i < 13; i++) {
//         int32_t j = i + 17;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 17] = W1[j] ^ W2[i];
//         Out[j + 34] = W2[j] ^ W3[i];
//         Out[i + 68] = W3[j] ^ W4[i];
//         Out[j + 68] = W4[j];
//     }
//     for (int32_t i = 13; i < 17; i++) {
//         int32_t j = i + 17;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 17] = W1[j] ^ W2[i];
//         Out[j + 34] = W2[j] ^ W3[i];
//         Out[i + 68] = W3[j] ^ W4[i];
//     }
//     Out[51] ^= W1[34];
//     Out[68] ^= W2[34];
// }

//len = 49: 2-Karatsuba
void gfmul_49_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[50], D1[50], D2[48], SAA[25], SBB[25];
    gfmul_25_vpclmul(D0, A, B);
    gfmul_24_vpclmul(D2, (A+25), (B+25));
    for(int32_t i = 0; i < 24; i++) {
        int32_t is = i + 25;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[24]=A[24];        SBB[24]=B[24];    gfmul_25_vpclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 23; i++) {
        int32_t is = i + 25;
        int32_t is2 = is + 25;
        int32_t is3 = is2 + 25;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 23; i < 25; i++) {
        int32_t is = i + 25;
        int32_t is2 = is + 25;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 96: TC3_128
void gfmul_96_vpclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[64], W1[66], W2[66], W3[66], W4[64];
    static __m256i tmp[66];
    __m128i zero128;
    zero128 = _mm_setzero_si128();
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[32];
    U2 = (const __m256i *)&A256[64];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[32];
    V2 = (const __m256i *)&B256[64];
    for (int32_t i = 0 ; i < 32 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_32_vpclmul(W1, W2, W3);
    const __m128i *U1_128 = ((const __m128i *) U1);
    const __m128i *V1_128 = ((const __m128i *) V1);
    W0[0] = _mm256_set_m128i(U1_128[0],zero128);
    W4[0] = _mm256_set_m128i(V1_128[0],zero128);
    U1_128 = ((const __m128i *) U1) + 1;
    V1_128 = ((const __m128i *) V1) + 1;
    for(int32_t i = 0; i < 31; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_set_m128i(V1_128[i2+1],V1_128[i2]);
        W4[i1] ^= V2[i];
    }
    W0[32] = _mm256_set_m128i(zero128,U1_128[62]) ^ U2[31];
    W4[32] = _mm256_set_m128i(zero128,V1_128[62]) ^ V2[31];
    for (int32_t i = 0 ; i < 32 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[32] = W0[32];
    W2[32] = W4[32];
    for (int32_t i = 0 ; i < 32 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_33_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 66; i++) {
        W3[i] = tmp[i];
    }
    gfmul_33_vpclmul(W2, W0, W4);
    gfmul_32_vpclmul(W4, U2, V2);
    gfmul_32_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 66 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 64 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_128 = ((const __m128i *) W2) + 1;
    const __m128i * U2_128 = ((const __m128i *) W0) + 1;
    for(int32_t i = 0; i < 63; i++) {
        int32_t i2 = i << 1;
        W2[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W2[i] ^= _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    W2[63]=_mm256_set_m128i(U1_128[127],U2_128[126]^U1_128[126]);
    W2[64]=_mm256_set_m128i(U1_128[129],U1_128[128]);
    W2[65]=_mm256_set_m128i(zero128,U1_128[130]);
    U1_128 = ((const __m128i *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ _mm256_set_m128i(U1_128[0],zero128);
    U1_128 = ((const __m128i *) W4) + 1;
    for(int32_t i = 2; i < 64; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^_mm256_set_m128i(U1_128[i2 - 3],U1_128[i2 - 4]);
    }
    tmp[64] = W2[64] ^ W3[64] ^ _mm256_set_m128i(U1_128[125],U1_128[124]);
    tmp[65] = W2[65] ^ W3[65] ^ _mm256_set_m128i(zero128,U1_128[126]);
    divide_by_x_plus_one_128(W2, tmp, 132);
    U1_128 = ((const __m128i *) W3) + 1;
    U2_128 = ((const __m128i *) W1) + 1;
    for(int32_t i = 0; i < 63; i++) {
        int32_t i2 = i << 1;
        tmp[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]) ^ _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    tmp[63]=_mm256_set_m128i(U1_128[127], U2_128[126]^U1_128[126]);
    tmp[64]=_mm256_set_m128i(U1_128[129],U1_128[128]);
    tmp[65]=_mm256_set_m128i(zero128, U1_128[130]);
    divide_by_x_plus_one_128(W3, tmp, 131);
    for (int32_t i = 0 ; i < 64 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[64] = W2[64];
    W1[65] = W2[65];
    for (int32_t i = 0 ; i < 65 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 32; i++)
    {
        int32_t j = i + 32;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 32] = W1[j] ^ W2[i];
        Out[j + 64] = W2[j] ^ W3[i];
        Out[i + 128] = W3[j] ^ W4[i];
        Out[j + 128] = W4[j];
    }
    Out[96] ^= W1[64];
    Out[97] ^= W1[65];
    Out[128] ^= W2[64];
    Out[129] ^= W2[65];
    Out[160] ^= W3[64];
}

//len = 97: TC3_256
void gfmul_97_vpclmul(__m256i *Out,  const  __m256i *A256,  const  __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[66], W1[67], W2[68], W3[68], W4[62], tmp[68];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[33];
    U2 = (const __m256i *)&A256[66];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[33];
    V2 = (const __m256i *)&B256[66];
    for (int32_t i = 0 ; i < 31 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[31] = U0[31] ^ U1[31];
    W2[31] = V0[31] ^ V1[31];
    W3[32] = U0[32] ^ U1[32];
    W2[32] = V0[32] ^ V1[32];
    gfmul_33_vpclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 32 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    
    W0[33] = U1[32];
    W4[33] = V1[32];
    for (int32_t i = 0 ; i < 33 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[33] = W0[33];
    W2[33] = W4[33];
    for (int32_t i = 0 ; i < 33 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_34_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 68 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_34_vpclmul(W2, W0, W4);
    gfmul_31_vpclmul(W4, U2, V2);
    gfmul_33_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 68 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 66 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 65 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[65] = W2[66];
    W2[66] = W2[67];
    for (int32_t i = 0 ; i < 62 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 62 ; i < 67 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[67] = W3[67];
    for (int32_t i = 0 ; i < 62 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 68);
    for (int32_t i = 0 ; i < 65 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[65] = W3[66];
    tmp[66] = W3[67];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 67);
    for (int32_t i = 0 ; i < 62 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 62 ; i < 66 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[66] = W2[66];
    for (int32_t i = 0 ; i < 66 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 29; i++) {
        int32_t j = i + 33;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 33] = W1[j] ^ W2[i];
        Out[j + 66] = W2[j] ^ W3[i];
        Out[i + 132] = W3[j] ^ W4[i];
        Out[j + 132] = W4[j];
    }
    for (int32_t i = 29; i < 33; i++) {
        int32_t j = i + 33;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 33] = W1[j] ^ W2[i];
        Out[j + 66] = W2[j] ^ W3[i];
        Out[i + 132] = W3[j] ^ W4[i];
    }
    Out[99] ^= W1[66];
    Out[132] ^= W2[66];
}

// //len = 160: 2-Karatsuba
// void gfmul_160_vpclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[160], D1[160], D2[160], SAA[80], SBB[80];
//     gfmul_80_vpclmul(D0, A, B);
//     gfmul_80_vpclmul(D2, (A+80), (B+80));
//     for(int32_t i = 0; i < 80; i++) {
//         int32_t is = i + 80;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//     gfmul_80_vpclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 80; i++) {
//         int32_t is = i + 80;
//         int32_t is2 = is + 80;
//         int32_t is3 = is2 + 80;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
// }

//len = 160: TC3_256
void gfmul_160_vpclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[108], W1[109], W2[110], W3[110], W4[104], tmp[110];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[54];
    U2 = (const __m256i *)&A256[108];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[54];
    V2 = (const __m256i *)&B256[108];
    for (int32_t i = 0 ; i < 52 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[52] = U0[52] ^ U1[52];
    W2[52] = V0[52] ^ V1[52];
    W3[53] = U0[53] ^ U1[53];
    W2[53] = V0[53] ^ V1[53];
    gfmul_54_vpclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 53 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    
    W0[54] = U1[53];
    W4[54] = V1[53];
    for (int32_t i = 0 ; i < 54 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[54] = W0[54];
    W2[54] = W4[54];
    for (int32_t i = 0 ; i < 54 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_55_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 110 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_55_vpclmul(W2, W0, W4);
    gfmul_52_vpclmul(W4, U2, V2);
    gfmul_54_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 110 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 108 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 107 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[107] = W2[108];
    W2[108] = W2[109];
    for (int32_t i = 0 ; i < 104 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 104 ; i < 109 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[109] = W3[109];
    for (int32_t i = 0 ; i < 104 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 110);
    for (int32_t i = 0 ; i < 107 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[107] = W3[108];
    tmp[108] = W3[109];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 109);
    for (int32_t i = 0 ; i < 104 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 104 ; i < 108 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[108] = W2[108];
    for (int32_t i = 0 ; i < 108 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 50; i++) {
        int32_t j = i + 54;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 54] = W1[j] ^ W2[i];
        Out[j + 108] = W2[j] ^ W3[i];
        Out[i + 216] = W3[j] ^ W4[i];
        Out[j + 216] = W4[j];
    }
    for (int32_t i = 50; i < 54; i++) {
        int32_t j = i + 54;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 54] = W1[j] ^ W2[i];
        Out[j + 108] = W2[j] ^ W3[i];
        Out[i + 216] = W3[j] ^ W4[i];
    }
    Out[162] ^= W1[108];
    Out[216] ^= W2[108];
}

//len = 161: TC3_256
void gfmul_161_vpclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[108], W1[109], W2[110], W3[110], W4[106], tmp[110];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[54];
    U2 = (const __m256i *)&A256[108];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[54];
    V2 = (const __m256i *)&B256[108];
    for (int32_t i = 0 ; i < 53 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[53] = U0[53] ^ U1[53];
    W2[53] = V0[53] ^ V1[53];
    gfmul_54_vpclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 54 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    for (int32_t i = 0 ; i < 54 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[54] = W0[54];
    W2[54] = W4[54];
    for (int32_t i = 0 ; i < 54 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_55_vpclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 110 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_55_vpclmul(W2, W0, W4);
    gfmul_53_vpclmul(W4, U2, V2);
    gfmul_54_vpclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 110 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 108 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 107 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[107] = W2[108];
    W2[108] = W2[109];
    for (int32_t i = 0 ; i < 106 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 106 ; i < 109 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[109] = W3[109];
    for (int32_t i = 0 ; i < 106 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 110);
    for (int32_t i = 0 ; i < 107 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[107] = W3[108];
    tmp[108] = W3[109];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 109);
    for (int32_t i = 0 ; i < 106 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 106 ; i < 108 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[108] = W2[108];
    for (int32_t i = 0 ; i < 108 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 52; i++) {
        int32_t j = i + 54;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 54] = W1[j] ^ W2[i];
        Out[j + 108] = W2[j] ^ W3[i];
        Out[i + 216] = W3[j] ^ W4[i];
        Out[j + 216] = W4[j];
    }
    for (int32_t i = 52; i < 54; i++) {
        int32_t j = i + 54;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 54] = W1[j] ^ W2[i];
        Out[j + 108] = W2[j] ^ W3[i];
        Out[i + 216] = W3[j] ^ W4[i];
    }
    Out[162] ^= W1[108];
    Out[216] ^= W2[108];
}

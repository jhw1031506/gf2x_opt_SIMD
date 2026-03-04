#include <immintrin.h>
#include <stdint.h>
#include <string.h>
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

static inline void divide_by_x_plus_one_64(__m256i *out, __m256i *in, int32_t size){
    uint64_t *A = (uint64_t*) in;
    uint64_t *B = (uint64_t*) out;

    B[0] = A[0];
    for(int32_t i = 1; i < size; i++) {
        B[i]= B[i - 1] ^ A[i];
    }
}
//len = 1: karat_karat_PCLMULQDQ
static inline void gfmul_1_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
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
static inline void gfmul_2_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[2], D1[2], D2[2], SAA[1], SBB[1];
    gfmul_1_pclmul(D0, A, B);
    gfmul_1_pclmul(D2, (A+1), (B+1));
    SAA[0] = A[0] ^ A[1];
    SBB[0] = B[0] ^ B[1];
    gfmul_1_pclmul(D1, SAA, SBB);
    __m256i middle = _mm256_xor_si256(D0[1], D2[0]);
    Out[0] = D0[0];
    Out[1] = middle ^ D0[0] ^ D1[0];
    Out[2] = middle ^ D1[1] ^ D2[1];
    Out[3] = D2[1];
}

// //len = 3: 2-Karatsuba
// static inline void gfmul_3_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[4], D1[4], D2[2], SAA[2], SBB[2];
//     gfmul_2_pclmul(D0, A, B);
//     gfmul_1_pclmul(D2, (A+2), (B+2));
//     SAA[0] = A[0] ^ A[2];
//     SBB[0] = B[0] ^ B[2];
//     SAA[1] = A[1];
//     SBB[1] = B[1];
//     gfmul_2_pclmul(D1, SAA, SBB);
//     __m256i middle = _mm256_xor_si256(D0[2], D2[0]);
//     Out[0]         = D0[0];
//     Out[2]         = middle ^ D0[0] ^ D1[0];
//     Out[4]         = middle ^ D1[2];
//     middle         = _mm256_xor_si256(D0[3], D2[1]);
//     Out[1]         = D0[1];
//     Out[3]         = middle ^ D0[1] ^ D1[1];
//     Out[5]         = middle ^ D1[3];
// }

//len = 3: 3-Karatsuba
static inline void gfmul_3_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
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
    gfmul_1_pclmul(D3, aa01, bb01);
    gfmul_1_pclmul(D4, aa02, bb02);
    gfmul_1_pclmul(D5, aa12, bb12);
    gfmul_1_pclmul(D0, a0, b0);
    gfmul_1_pclmul(D1, a1, b1);
    gfmul_1_pclmul(D2, a2, b2);
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
static inline void gfmul_4_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[4], D1[4], D2[4], SAA[2], SBB[2];
    gfmul_2_pclmul(D0, A, B);
    gfmul_2_pclmul(D2, (A+2), (B+2));
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 2;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_2_pclmul(D1, SAA, SBB);
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

// //len = 5: 2-Karatsuba
// static inline void gfmul_5_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[6], D1[6], D2[4], SAA[3], SBB[3];
//     gfmul_3_pclmul(D0, A, B);
//     gfmul_2_pclmul(D2, (A+3), (B+3));
//     for(int32_t i = 0; i < 2; i++) {
//         int32_t is = i + 3;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//     SAA[2]=A[2];        
//     SBB[2]=B[2];    
//     gfmul_3_pclmul(D1, SAA, SBB);
//     __m256i middle = _mm256_xor_si256(D0[3], D2[0]);
//     Out[0] = D0[0];
//     Out[3] = middle ^ D0[0] ^ D1[0];
//     Out[6] = middle ^ D1[3] ^ D2[3];
//     Out[9] = D2[3];
//     for(int32_t i = 1; i < 3; i++) {
//         int32_t is = i + 3;
//         int32_t is2 = is + 3;
//         middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is];
//     }
// }

//len = 5: 5-Karatsuba
static inline void gfmul_5_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
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
    gfmul_1_pclmul(D01, aa01, bb01);
    gfmul_1_pclmul(D02, aa02, bb02);
    gfmul_1_pclmul(D03, aa03, bb03);
    gfmul_1_pclmul(D04, aa04, bb04);
    gfmul_1_pclmul(D12, aa12, bb12);
    gfmul_1_pclmul(D13, aa13, bb13);
    gfmul_1_pclmul(D14, aa14, bb14);
    gfmul_1_pclmul(D23, aa23, bb23);
    gfmul_1_pclmul(D24, aa24, bb24);
    gfmul_1_pclmul(D34, aa34, bb34);
    gfmul_1_pclmul(D0, a0, b0);
    gfmul_1_pclmul(D1, a1, b1);
    gfmul_1_pclmul(D2, a2, b2);
    gfmul_1_pclmul(D3, a3, b3);
    gfmul_1_pclmul(D4, a4, b4);
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

// //len = 6: 3-Karatsuba
// static inline void gfmul_6_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
//     __m256i middle;
//     static __m256i aa01[2], bb01[2], aa02[2], bb02[2], aa12[2], bb12[2];
//     static __m256i D0[4], D1[4], D2[4], D3[4], D4[4], D5[4];
//     a0 = A;
//     a1 = A + 2;
//     a2 = A + 4;
//     b0 = B;
//     b1 = B + 2;
//     b2 = B + 4;
//     for (int16_t i = 0; i < 2; i++)
//     {
//         aa01[i] = a0[i] ^ a1[i];
//         bb01[i] = b0[i] ^ b1[i];
//         aa12[i] = a2[i] ^ a1[i];
//         bb12[i] = b2[i] ^ b1[i];
//         aa02[i] = a0[i] ^ a2[i];
//         bb02[i] = b0[i] ^ b2[i];
//     }
//     gfmul_2_pclmul(D3, aa01, bb01);
//     gfmul_2_pclmul(D4, aa02, bb02);
//     gfmul_2_pclmul(D5, aa12, bb12);
//     gfmul_2_pclmul(D0, a0, b0);
//     gfmul_2_pclmul(D1, a1, b1);
//     gfmul_2_pclmul(D2, a2, b2);
//     for (int16_t i = 0; i < 2; i++)
//     {
//         int16_t j = i + 2;
//         middle = D0[i] ^ D1[i] ^ D0[j];
//         Out[i] = D0[i];
//         Out[j] = D3[i] ^ middle;
//         Out[j + 2] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
//         middle = D1[j] ^ D2[i] ^ D2[j];
//         Out[j + 4] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
//         Out[i + 8] = D5[j] ^ middle;
//         Out[j + 8] = D2[j];
//     }
// }

//len = 6: 2-Karatsuba
static inline void gfmul_6_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[6], D1[6], D2[6], SAA[3], SBB[3];
    gfmul_3_pclmul(D0, A, B);
    gfmul_3_pclmul(D2, (A+3), (B+3));
    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 3;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_3_pclmul(D1, SAA, SBB);
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

// //len = 7: 2-Karatsuba
// static inline void gfmul_7_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[8], D1[8], D2[6], SAA[4], SBB[4];
//     gfmul_4_pclmul(D0, A, B);
//     gfmul_3_pclmul(D2, (A+4), (B+4));
//     for(int32_t i = 0; i < 3; i++) {
//         int32_t is = i + 4;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//         SAA[3]=A[3];        SBB[3]=B[3];    gfmul_4_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 2; i++) {
//         int32_t is = i + 4;
//         int32_t is2 = is + 4;
//         int32_t is3 = is2 + 4;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
//     for(int32_t i = 2; i < 4; i++) {
//         int32_t is = i + 4;
//         int32_t is2 = is + 4;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is];
//     }
// }

//len = 8: 2-Karatsuba
static inline void gfmul_8_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[8], D1[8], D2[8], SAA[4], SBB[4];
    gfmul_4_pclmul(D0, A, B);
    gfmul_4_pclmul(D2, (A+4), (B+4));
    for(int32_t i = 0; i < 4; i++) {
        int32_t is = i + 4;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_4_pclmul(D1, SAA, SBB);
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

// //len = 9: 2-Karatsuba
// static inline void gfmul_9_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[10], D1[10], D2[8], SAA[5], SBB[5];
//     gfmul_5_pclmul(D0, A, B);
//     gfmul_4_pclmul(D2, (A+5), (B+5));
//     for(int32_t i = 0; i < 4; i++) {
//         int32_t is = i + 5;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//         SAA[4]=A[4];        SBB[4]=B[4];    gfmul_5_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 3; i++) {
//         int32_t is = i + 5;
//         int32_t is2 = is + 5;
//         int32_t is3 = is2 + 5;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
//     for(int32_t i = 3; i < 5; i++) {
//         int32_t is = i + 5;
//         int32_t is2 = is + 5;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is];
//     }
// }

//len = 9: 3-Karatsuba
static inline void gfmul_9_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
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
    gfmul_3_pclmul(D3, aa01, bb01);
    gfmul_3_pclmul(D4, aa02, bb02);
    gfmul_3_pclmul(D5, aa12, bb12);
    gfmul_3_pclmul(D0, a0, b0);
    gfmul_3_pclmul(D1, a1, b1);
    gfmul_3_pclmul(D2, a2, b2);
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
static inline void gfmul_10_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[10], D1[10], D2[10], SAA[5], SBB[5];
    gfmul_5_pclmul(D0, A, B);
    gfmul_5_pclmul(D2, (A+5), (B+5));
    for(int32_t i = 0; i < 5; i++) {
        int32_t is = i + 5;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_5_pclmul(D1, SAA, SBB);
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

//len = 11: 2-Karatsuba
static inline void gfmul_11_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[12], D1[12], D2[10], SAA[6], SBB[6];
    gfmul_6_pclmul(D0, A, B);
    gfmul_5_pclmul(D2, (A+6), (B+6));
    for(int32_t i = 0; i < 5; i++) {
        int32_t is = i + 6;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[5]=A[5];        SBB[5]=B[5];    gfmul_6_pclmul(D1, SAA, SBB);
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

// //len = 12: 3-Karatsuba
// static inline void gfmul_12_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
//     __m256i middle;
//     static __m256i aa01[4], bb01[4], aa02[4], bb02[4], aa12[4], bb12[4];
//     static __m256i D0[8], D1[8], D2[8], D3[8], D4[8], D5[8];
//     a0 = A;
//     a1 = A + 4;
//     a2 = A + 8;
//     b0 = B;
//     b1 = B + 4;
//     b2 = B + 8;
//     for (int16_t i = 0; i < 4; i++)
//     {
//         aa01[i] = a0[i] ^ a1[i];
//         bb01[i] = b0[i] ^ b1[i];
//         aa12[i] = a2[i] ^ a1[i];
//         bb12[i] = b2[i] ^ b1[i];
//         aa02[i] = a0[i] ^ a2[i];
//         bb02[i] = b0[i] ^ b2[i];
//     }
//     gfmul_4_pclmul(D3, aa01, bb01);
//     gfmul_4_pclmul(D4, aa02, bb02);
//     gfmul_4_pclmul(D5, aa12, bb12);
//     gfmul_4_pclmul(D0, a0, b0);
//     gfmul_4_pclmul(D1, a1, b1);
//     gfmul_4_pclmul(D2, a2, b2);
//     for (int16_t i = 0; i < 4; i++)
//     {
//         int16_t j = i + 4;
//         middle = D0[i] ^ D1[i] ^ D0[j];
//         Out[i] = D0[i];
//         Out[j] = D3[i] ^ middle;
//         Out[j + 4] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
//         middle = D1[j] ^ D2[i] ^ D2[j];
//         Out[j + 8] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
//         Out[i + 16] = D5[j] ^ middle;
//         Out[j + 16] = D2[j];
//     }
// }

//len = 12: 2-Karatsuba
static inline void gfmul_12_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[12], D1[12], D2[12], SAA[6], SBB[6];
    gfmul_6_pclmul(D0, A, B);
    gfmul_6_pclmul(D2, (A+6), (B+6));
    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 6;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_6_pclmul(D1, SAA, SBB);
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

// //len = 13: 2-Karatsuba
// static inline void gfmul_13_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[14], D1[14], D2[12], SAA[7], SBB[7];
//     gfmul_7_pclmul(D0, A, B);
//     gfmul_6_pclmul(D2, (A+7), (B+7));
//     for(int32_t i = 0; i < 6; i++) {
//         int32_t is = i + 7;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//         SAA[6]=A[6];        SBB[6]=B[6];    gfmul_7_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 5; i++) {
//         int32_t is = i + 7;
//         int32_t is2 = is + 7;
//         int32_t is3 = is2 + 7;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
//     for(int32_t i = 5; i < 7; i++) {
//         int32_t is = i + 7;
//         int32_t is2 = is + 7;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is];
//     }
// }

// //len = 14: 2-Karatsuba
// static inline void gfmul_14_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[14], D1[14], D2[14], SAA[7], SBB[7];
//     gfmul_7_pclmul(D0, A, B);
//     gfmul_7_pclmul(D2, (A+7), (B+7));
//     for(int32_t i = 0; i < 7; i++) {
//         int32_t is = i + 7;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//     gfmul_7_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 7; i++) {
//         int32_t is = i + 7;
//         int32_t is2 = is + 7;
//         int32_t is3 = is2 + 7;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
// }

// //len = 15: 2-Karatsuba
// static inline void gfmul_15_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[16], D1[16], D2[14], SAA[8], SBB[8];
//     gfmul_8_pclmul(D0, A, B);
//     gfmul_7_pclmul(D2, (A+8), (B+8));
//     for(int32_t i = 0; i < 7; i++) {
//         int32_t is = i + 8;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//         SAA[7]=A[7];        SBB[7]=B[7];    gfmul_8_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 6; i++) {
//         int32_t is = i + 8;
//         int32_t is2 = is + 8;
//         int32_t is3 = is2 + 8;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
//     for(int32_t i = 6; i < 8; i++) {
//         int32_t is = i + 8;
//         int32_t is2 = is + 8;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is];
//     }
// }

//len = 16: 2-Karatsuba
static inline void gfmul_16_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[16], D1[16], D2[16], SAA[8], SBB[8];
    gfmul_8_pclmul(D0, A, B);
    gfmul_8_pclmul(D2, (A+8), (B+8));
    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 8;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_8_pclmul(D1, SAA, SBB);
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

//len = 17: 2-Karatsuba
static inline void gfmul_17_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[18], D1[18], D2[16], SAA[9], SBB[9];
    gfmul_9_pclmul(D0, A, B);
    gfmul_8_pclmul(D2, (A+9), (B+9));
    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 9;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[8]=A[8];        SBB[8]=B[8];    gfmul_9_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 9;
        int32_t is2 = is + 9;
        int32_t is3 = is2 + 9;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 7; i < 9; i++) {
        int32_t is = i + 9;
        int32_t is2 = is + 9;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

// //len = 18: TC3_256
// static inline void gfmul_18_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
//     const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
//     static __m256i W0[12], W1[15], W2[16], W3[16], W4[12], tmp[16];
//     static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
//     U0 = (const __m256i *)&A256[0];
//     U1 = (const __m256i *)&A256[6];
//     U2 = (const __m256i *)&A256[12];
//     V0 = (const __m256i *)&B256[0];
//     V1 = (const __m256i *)&B256[6];
//     V2 = (const __m256i *)&B256[12];
//     for (int32_t i = 0 ; i < 6 ; i++) {
//         W3[i] = U0[i] ^ U1[i] ^ U2[i];
//         W2[i] = V0[i] ^ V1[i] ^ V2[i];
//     }
//     gfmul_6_pclmul(W1, W2, W3);
//     W0[0] = zero;
//     W4[0] = zero;
//     W0[1] = U1[0];
//     W4[1] = V1[0];
//     for (int32_t i = 1 ; i < 6 ; i++) {
//         W0[i + 1] = U1[i] ^ U2[i - 1];
//         W4[i + 1] = V1[i] ^ V2[i - 1];
//     }
//     W0[7] = U2[5];
//     W4[7] = V2[5];
//     for (int32_t i = 0 ; i < 6 ; i++) {
//         W3[i] ^= W0[i];
//         W2[i] ^= W4[i];
//     }
//     W3[6] = W0[6];
//     W3[7] = W0[7];
//     W2[6] = W4[6];
//     W2[7] = W4[7];
//     for (int32_t i = 0 ; i < 6 ; i++) {
//         W0[i] ^= U0[i];
//         W4[i] ^= V0[i];
//     }
//     gfmul_8_pclmul(tmp, W3, W2);
//     for (int32_t i = 0 ; i < 16 ; i++) {
//         W3[i] = tmp[i];
//     }
//     gfmul_8_pclmul(W2, W0, W4);
//     gfmul_6_pclmul(W4, U2, V2);
//     gfmul_6_pclmul(W0, U0, V0);
//     for (int32_t i = 0 ; i < 16 ; i++) {
//         W3[i] ^= W2[i];
//     }
//     for (int32_t i = 0 ; i < 12 ; i++) {
//         W1[i] ^= W0[i];
//     }
//     for (int32_t i = 0 ; i < 11 ; i++) {
//         int32_t i1 = i + 1;
//         W2[i] = W2[i1] ^ W0[i1];
//     }
//     W2[11] = W2[12];
//     W2[12] = W2[13];
//     W2[13] = W2[14];
//     W2[14] = W2[15];
//     for (int32_t i = 0 ; i < 12 ; i++) {
//         tmp[i] = W2[i] ^ W3[i] ^ W4[i];
//     }
//     for (int32_t i = 12 ; i < 15 ; i++) {
//         tmp[i] = W2[i] ^ W3[i];
//     }
//     tmp[15] = W3[15];
//     for (int32_t i = 0 ; i < 12 ; i++) {
//         tmp[i + 3] ^= W4[i];
//     }
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 16);
//     for (int32_t i = 0 ; i < 11 ; i++) {
//         int32_t i1 = i + 1;
//         tmp[i] = W3[i1] ^ W1[i1];
//     }
//     tmp[11] = W3[12];
//     tmp[12] = W3[13];
//     tmp[13] = W3[14];
//     tmp[14] = W3[15];
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 15);
//     for (int32_t i = 0 ; i < 12 ; i++) {
//         W1[i] ^= W2[i] ^ W4[i];
//     }
//     for (int32_t i = 12 ; i < 15 ; i++) {
//         W1[i] = W2[i];
//     }
//     for (int32_t i = 0 ; i < 14 ; i++) {
//         W2[i] ^= W3[i];
//     }
//     for (int32_t i = 0; i < 6; i++) {
//         int32_t j = i + 6;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 6] = W1[j] ^ W2[i];
//         Out[j + 12] = W2[j] ^ W3[i];
//         Out[i + 24] = W3[j] ^ W4[i];
//         Out[j + 24] = W4[j];
//     }
//     Out[18] ^= W1[12];
//     Out[19] ^= W1[13];
//     Out[20] ^= W1[14];
//     Out[24] ^= W2[12];
//     Out[25] ^= W2[13];
//     Out[26] ^= W2[14];
//     Out[30] ^= W3[12];
//     Out[31] ^= W3[13];
// }

//len = 18: 2-Karatsuba
static inline void gfmul_18_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[18], D1[18], D2[18], SAA[9], SBB[9];
    gfmul_9_pclmul(D0, A, B);
    gfmul_9_pclmul(D2, (A+9), (B+9));
    for(int32_t i = 0; i < 9; i++) {
        int32_t is = i + 9;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_9_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 9; i++) {
        int32_t is = i + 9;
        int32_t is2 = is + 9;
        int32_t is3 = is2 + 9;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

// //len = 19: TC3_256
// static inline void gfmul_19_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
//     const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
//     static __m256i W0[14], W1[15], W2[16], W3[16], W4[10], tmp[16];
//     static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
//     U0 = (const __m256i *)&A256[0];
//     U1 = (const __m256i *)&A256[7];
//     U2 = (const __m256i *)&A256[14];
//     V0 = (const __m256i *)&B256[0];
//     V1 = (const __m256i *)&B256[7];
//     V2 = (const __m256i *)&B256[14];
//     for (int32_t i = 0 ; i < 5 ; i++) {
//         W3[i] = U0[i] ^ U1[i] ^ U2[i];
//         W2[i] = V0[i] ^ V1[i] ^ V2[i];
//     }
//     W3[5] = U0[5] ^ U1[5];
//     W2[5] = V0[5] ^ V1[5];
//     W3[6] = U0[6] ^ U1[6];
//     W2[6] = V0[6] ^ V1[6];
//     gfmul_7_pclmul(W1, W2, W3);
//     W0[0] = zero;
//     W4[0] = zero;
//     W0[1] = U1[0];
//     W4[1] = V1[0];
//     for (int32_t i = 1 ; i < 6 ; i++) {
//         W0[i + 1] = U1[i] ^ U2[i - 1];
//         W4[i + 1] = V1[i] ^ V2[i - 1];
//     }
//     W0[7] = U1[6];
//     W4[7] = V1[6];
//     for (int32_t i = 0 ; i < 7 ; i++) {
//         W3[i] ^= W0[i];
//         W2[i] ^= W4[i];
//     }
//     W3[7] = W0[7];
//     W2[7] = W4[7];
//     for (int32_t i = 0 ; i < 7 ; i++) {
//         W0[i] ^= U0[i];
//         W4[i] ^= V0[i];
//     }
//     gfmul_8_pclmul(tmp, W3, W2);
//     for (int32_t i = 0 ; i < 16 ; i++) {
//         W3[i] = tmp[i];
//     }
//     gfmul_8_pclmul(W2, W0, W4);
//     gfmul_5_pclmul(W4, U2, V2);
//     gfmul_7_pclmul(W0, U0, V0);
//     for (int32_t i = 0 ; i < 16 ; i++) {
//         W3[i] ^= W2[i];
//     }
//     for (int32_t i = 0 ; i < 14 ; i++) {
//         W1[i] ^= W0[i];
//     }
//     for (int32_t i = 0 ; i < 13 ; i++) {
//         int32_t i1 = i + 1;
//         W2[i] = W2[i1] ^ W0[i1];
//     }
//     W2[13] = W2[14];
//     W2[14] = W2[15];
//     for (int32_t i = 0 ; i < 10 ; i++) {
//         tmp[i] = W2[i] ^ W3[i] ^ W4[i];
//     }
//     for (int32_t i = 10 ; i < 15 ; i++) {
//         tmp[i] = W2[i] ^ W3[i];
//     }
//     tmp[15] = W3[15];
//     for (int32_t i = 0 ; i < 10 ; i++) {
//         tmp[i + 3] ^= W4[i];
//     }
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 16);
//     for (int32_t i = 0 ; i < 13 ; i++) {
//         int32_t i1 = i + 1;
//         tmp[i] = W3[i1] ^ W1[i1];
//     }
//     tmp[13] = W3[14];
//     tmp[14] = W3[15];
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 15);
//     for (int32_t i = 0 ; i < 10 ; i++) {
//         W1[i] ^= W2[i] ^ W4[i];
//     }
//     for (int32_t i = 10 ; i < 14 ; i++) {
//         W1[i] ^= W2[i];
//     }
//     W1[14] = W2[14];
//     for (int32_t i = 0 ; i < 14 ; i++) {
//         W2[i] ^= W3[i];
//     }
//     for (int32_t i = 0; i < 3; i++) {
//         int32_t j = i + 7;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 7] = W1[j] ^ W2[i];
//         Out[j + 14] = W2[j] ^ W3[i];
//         Out[i + 28] = W3[j] ^ W4[i];
//         Out[j + 28] = W4[j];
//     }
//     for (int32_t i = 3; i < 7; i++) {
//         int32_t j = i + 7;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 7] = W1[j] ^ W2[i];
//         Out[j + 14] = W2[j] ^ W3[i];
//         Out[i + 28] = W3[j] ^ W4[i];
//     }
//     Out[21] ^= W1[14];
//     Out[28] ^= W2[14];
// }

//len = 19: 2-Karatsuba
static inline void gfmul_19_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[20], D1[20], D2[18], SAA[10], SBB[10];
    gfmul_10_pclmul(D0, A, B);
    gfmul_9_pclmul(D2, (A+10), (B+10));
    for(int32_t i = 0; i < 9; i++) {
        int32_t is = i + 10;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[9]=A[9];        SBB[9]=B[9];    gfmul_10_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 10;
        int32_t is2 = is + 10;
        int32_t is3 = is2 + 10;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 8; i < 10; i++) {
        int32_t is = i + 10;
        int32_t is2 = is + 10;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

// //len = 24: 3-Karatsuba
// static inline void gfmul_24_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
//     static __m256i middle;
//     static __m256i aa01[8], bb01[8], aa02[8], bb02[8], aa12[8], bb12[8];
//     static __m256i D0[16], D1[16], D2[16], D3[16], D4[16], D5[16];
//     a0 = A;
//     a1 = A + 8;
//     a2 = A + 16;
//     b0 = B;
//     b1 = B + 8;
//     b2 = B + 16;
//     for (int16_t i = 0; i < 8; i++)
//     {
//         aa01[i] = a0[i] ^ a1[i];
//         bb01[i] = b0[i] ^ b1[i];
//         aa12[i] = a2[i] ^ a1[i];
//         bb12[i] = b2[i] ^ b1[i];
//         aa02[i] = a0[i] ^ a2[i];
//         bb02[i] = b0[i] ^ b2[i];
//     }
//     gfmul_8_pclmul(D3, aa01, bb01);
//     gfmul_8_pclmul(D4, aa02, bb02);
//     gfmul_8_pclmul(D5, aa12, bb12);
//     gfmul_8_pclmul(D0, a0, b0);
//     gfmul_8_pclmul(D1, a1, b1);
//     gfmul_8_pclmul(D2, a2, b2);
//     for (int16_t i = 0; i < 8; i++)
//     {
//         int16_t j = i + 8;
//         middle = D0[i] ^ D1[i] ^ D0[j];
//         Out[i] = D0[i];
//         Out[j] = D3[i] ^ middle;
//         Out[j + 8] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
//         middle = D1[j] ^ D2[i] ^ D2[j];
//         Out[j + 16] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
//         Out[i + 32] = D5[j] ^ middle;
//         Out[j + 32] = D2[j];
//     }
// }

//len = 24: 2-Karatsuba
static inline void gfmul_24_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[24], D1[24], D2[24], SAA[12], SBB[12];
    gfmul_12_pclmul(D0, A, B);
    gfmul_12_pclmul(D2, (A+12), (B+12));
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 12;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_12_pclmul(D1, SAA, SBB);
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

// //len = 25: 2-Karatsuba
// static inline void gfmul_25_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[26], D1[26], D2[24], SAA[13], SBB[13];
//     gfmul_13_pclmul(D0, A, B);
//     gfmul_12_pclmul(D2, (A+13), (B+13));
//     for(int32_t i = 0; i < 12; i++) {
//         int32_t is = i + 13;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//         SAA[12]=A[12];        SBB[12]=B[12];    gfmul_13_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 11; i++) {
//         int32_t is = i + 13;
//         int32_t is2 = is + 13;
//         int32_t is3 = is2 + 13;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
//     for(int32_t i = 11; i < 13; i++) {
//         int32_t is = i + 13;
//         int32_t is2 = is + 13;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is];
//     }
// }

// //len = 31: 2-Karatsuba
// static inline void gfmul_31_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[32], D1[32], D2[30], SAA[16], SBB[16];
//     gfmul_16_pclmul(D0, A, B);
//     gfmul_15_pclmul(D2, (A+16), (B+16));
//     for(int32_t i = 0; i < 15; i++) {
//         int32_t is = i + 16;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//         SAA[15]=A[15];        SBB[15]=B[15];    gfmul_16_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 14; i++) {
//         int32_t is = i + 16;
//         int32_t is2 = is + 16;
//         int32_t is3 = is2 + 16;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
//     for(int32_t i = 14; i < 16; i++) {
//         int32_t is = i + 16;
//         int32_t is2 = is + 16;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is];
//     }
// }

// //len = 32: 2-Karatsuba
// static inline void gfmul_32_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[32], D1[32], D2[32], SAA[16], SBB[16];
//     gfmul_16_pclmul(D0, A, B);
//     gfmul_16_pclmul(D2, (A+16), (B+16));
//     for(int32_t i = 0; i < 16; i++) {
//         int32_t is = i + 16;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//     gfmul_16_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 16; i++) {
//         int32_t is = i + 16;
//         int32_t is2 = is + 16;
//         int32_t is3 = is2 + 16;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
// }

//len = 32: TC3_256
static inline void gfmul_32_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[22], W1[23], W2[24], W3[24], W4[20], tmp[24];
    static __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[11];
    U2 = (const __m256i *)&A256[22];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[11];
    V2 = (const __m256i *)&B256[22];
    for (int32_t i = 0 ; i < 10 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[10] = U0[10] ^ U1[10];
    W2[10] = V0[10] ^ V1[10];
    gfmul_11_pclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 11 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
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
    gfmul_12_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 24 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_12_pclmul(W2, W0, W4);
    gfmul_10_pclmul(W4, U2, V2);
    gfmul_11_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 24 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 22 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 21 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[21] = W2[22];
    W2[22] = W2[23];
    for (int32_t i = 0 ; i < 20 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 20 ; i < 23 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[23] = W3[23];
    for (int32_t i = 0 ; i < 20 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 24);
    for (int32_t i = 0 ; i < 21 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[21] = W3[22];
    tmp[22] = W3[23];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 23);
    for (int32_t i = 0 ; i < 20 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 20 ; i < 22 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[22] = W2[22];
    for (int32_t i = 0 ; i < 22 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 9; i++) {
        int32_t j = i + 11;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 11] = W1[j] ^ W2[i];
        Out[j + 22] = W2[j] ^ W3[i];
        Out[i + 44] = W3[j] ^ W4[i];
        Out[j + 44] = W4[j];
    }
    for (int32_t i = 9; i < 11; i++) {
        int32_t j = i + 11;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 11] = W1[j] ^ W2[i];
        Out[j + 22] = W2[j] ^ W3[i];
        Out[i + 44] = W3[j] ^ W4[i];
    }
    Out[33] ^= W1[22];
    Out[44] ^= W2[22];
}

// //len = 33: 2-Karatsuba
// static inline void gfmul_33_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[34], D1[34], D2[32], SAA[17], SBB[17];
//     gfmul_17_pclmul(D0, A, B);
//     gfmul_16_pclmul(D2, (A+17), (B+17));
//     for(int32_t i = 0; i < 16; i++) {
//         int32_t is = i + 17;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//         SAA[16]=A[16];        SBB[16]=B[16];    gfmul_17_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 15; i++) {
//         int32_t is = i + 17;
//         int32_t is2 = is + 17;
//         int32_t is3 = is2 + 17;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
//     for(int32_t i = 15; i < 17; i++) {
//         int32_t is = i + 17;
//         int32_t is2 = is + 17;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is];
//     }
// }

//len = 33: TC3_128
static inline void gfmul_33_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
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
    gfmul_11_pclmul(W1, W2, W3);
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
    gfmul_12_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 24; i++) {
        W3[i] = tmp[i];
    }
    gfmul_12_pclmul(W2, W0, W4);
    gfmul_11_pclmul(W4, U2, V2);
    gfmul_11_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 24 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 22 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_128 = ((const __m128i *) W2) + 1;
    const __m128i * U2_128 = ((const __m128i *) W0) + 1;
    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        W2[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W2[i] ^= _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    W2[21]=_mm256_set_m128i(U1_128[43],U2_128[42]^U1_128[42]);
    W2[22]=_mm256_set_m128i(U1_128[45],U1_128[44]);
    W2[23]=_mm256_set_m128i(zero128,U1_128[46]);
    U1_128 = ((const __m128i *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ _mm256_set_m128i(U1_128[0],zero128);
    U1_128 = ((const __m128i *) W4) + 1;
    for(int32_t i = 2; i < 22; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^_mm256_set_m128i(U1_128[i2 - 3],U1_128[i2 - 4]);
    }
    tmp[22] = W2[22] ^ W3[22] ^ _mm256_set_m128i(U1_128[41],U1_128[40]);
    tmp[23] = W2[23] ^ W3[23] ^ _mm256_set_m128i(zero128,U1_128[42]);
    divide_by_x_plus_one_128(W2, tmp, 48);
    U1_128 = ((const __m128i *) W3) + 1;
    U2_128 = ((const __m128i *) W1) + 1;
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

// //len = 34: TC3_128
// static inline void gfmul_34_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
//     static __m256i U0[12], U1[12], U2[11], V0[12], V1[12], V2[11];
//     static __m256i W0[24], W1[24], W2[24], W3[25], W4[22];
//     static __m256i tmp[25];
//     const uint64_t *A = (const uint64_t *) A256;
//     const uint64_t *B = (const uint64_t *) B256;
//     for(int32_t i = 0; i < 11; i++) {
//         int32_t i4 = i << 2;
//         U0[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4]));
//         V0[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4]));
//         U1[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 46]));
//         V1[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 46]));
//         U2[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 92]));
//         V2[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 92]));
//     }
//     U0[11]= (__m256i){A[44], A[45], 0x0ul, 0x0ul};
//     V0[11]= (__m256i){B[44], B[45], 0x0ul, 0x0ul};
//     U1[11]= (__m256i){A[90], A[91], 0x0ul, 0x0ul};
//     V1[11]= (__m256i){B[90], B[91], 0x0ul, 0x0ul};
//     for (int32_t i = 0 ; i < 11 ; i++) {
//         W3[i] = U0[i] ^ U1[i] ^ U2[i];
//         W2[i] = V0[i] ^ V1[i] ^ V2[i];
//     }
//     W3[11] = U0[11] ^ U1[11];
//     W2[11] = V0[11] ^ V1[11];
//     gfmul_12_pclmul(W1, W2, W3);
//     uint64_t *U1_64 = ((uint64_t *) U1);
//     uint64_t *V1_64 = ((uint64_t *) V1);
//     W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};
//     W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};
//     U1_64 = ((uint64_t *) U1) + 2;
//     V1_64 = ((uint64_t *) V1) + 2;
//     for(int32_t i = 0; i < 11; i++) {
//         int32_t i4 = i << 2;
//         int32_t i1 = i + 1;
//         W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
//         W0[i1] ^= U2[i];
//         W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));
//         W4[i1] ^= V2[i];
//     }
//     for (int32_t i = 0 ; i < 12 ; i++) {
//         W3[i] ^= W0[i];
//         W2[i] ^= W4[i];
//     }
//     for (int32_t i = 0 ; i < 12 ; i++) {
//         W0[i] ^= U0[i];
//         W4[i] ^= V0[i];
//     }
//     gfmul_12_pclmul(tmp, W3, W2);
//     for (int32_t i = 0 ; i < 24; i++) {
//         W3[i] = tmp[i];
//     }
//     gfmul_12_pclmul(W2, W0, W4);
//     gfmul_11_pclmul(W4, U2, V2);
//     gfmul_12_pclmul(W0, U0, V0);
//     for (int32_t i = 0 ; i < 24 ; i++) {
//         W3[i] ^= W2[i];
//     }
//     for (int32_t i = 0 ; i < 23 ; i++) {
//         W1[i] ^= W0[i];
//     }
//     U1_64 = ((uint64_t *) W2) + 2;
//     uint64_t * U2_64 = ((uint64_t *) W0) + 2;
//     for(int32_t i = 0; i < 22; i++) {
//         int32_t i4 = i << 2;
//         W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
//         W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
//     }
//     W2[22]=(__m256i){U2_64[88], U2_64[89], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[88]));
//     W2[23]=(__m256i){U1_64[92], U1_64[93], 0ul, 0ul};
//     U1_64 = ((uint64_t *) W4);
//     tmp[0] = W2[0] ^ W3[0] ^ W4[0];
//     tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};
//     U1_64 = ((uint64_t *) W4) + 2;
//     for(int32_t i = 2; i < 22; i++) {
//         int32_t i4 = i << 2;
//         tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));
//     }
//     tmp[22] = W2[22] ^ W3[22] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[80]));
//     tmp[23] = W2[23] ^ W3[23] ^ (__m256i){U1_64[84],U1_64[85], 0ul, 0ul};
//     divide_by_x_plus_one_128(W2, tmp, 48);
//     U1_64 = ((uint64_t *) W3) + 2;
//     U2_64 = ((uint64_t *) W1) + 2;
//     for(int32_t i = 0; i < 22; i++) {
//         int32_t i4 = i << 2;
//         tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
//     }
//     tmp[22]=(__m256i){U2_64[88], U2_64[89], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[88]));
//     tmp[23]=(__m256i){U1_64[92], U1_64[93], 0ul, 0ul};
//     divide_by_x_plus_one_128(W3, tmp, 47);
//     for (int32_t i = 0 ; i < 22 ; i++) {
//         W1[i] ^= W2[i] ^ W4[i];
//     }
//     W1[22] ^= W2[22];
//     W1[23] = W2[23];
//     for (int32_t i = 0 ; i < 23 ; i++) {
//         W2[i] ^= W3[i];
//     }
//     for(int32_t i = 0; i < 22; i++) {
//         Out[i] = W0[i];
//         Out[i + 23] = W2[i];
//         Out[i + 46] = W4[i];
//     }
//     Out[22] = W0[22];
//     Out[45] = W2[22];
//     Out[46] ^= W2[23];
//     U1_64 = ((uint64_t *) &Out[11]) + 2;
//     U2_64 = ((uint64_t *) &Out[34]) + 2;
//     __m256i aux;
//     for(int32_t i = 0; i < 24; i++) {
//         int32_t i4 = i << 2;
//         aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];
//         _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);
//         aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W3[i];
//         _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);
//     }
// }

// //len = 40: TC3_256
// static inline void gfmul_40_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
//     const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
//     static __m256i W0[28], W1[29], W2[30], W3[30], W4[24], tmp[30];
//     static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
//     U0 = (const __m256i *)&A256[0];
//     U1 = (const __m256i *)&A256[14];
//     U2 = (const __m256i *)&A256[28];
//     V0 = (const __m256i *)&B256[0];
//     V1 = (const __m256i *)&B256[14];
//     V2 = (const __m256i *)&B256[28];
//     for (int32_t i = 0 ; i < 12 ; i++) {
//         W3[i] = U0[i] ^ U1[i] ^ U2[i];
//         W2[i] = V0[i] ^ V1[i] ^ V2[i];
//     }
//     W3[12] = U0[12] ^ U1[12];
//     W2[12] = V0[12] ^ V1[12];
//     W3[13] = U0[13] ^ U1[13];
//     W2[13] = V0[13] ^ V1[13];
//     gfmul_14_pclmul(W1, W2, W3);
//     W0[0] = zero;
//     W4[0] = zero;
//     W0[1] = U1[0];
//     W4[1] = V1[0];
//     for (int32_t i = 1 ; i < 13 ; i++) {
//         W0[i + 1] = U1[i] ^ U2[i - 1];
//         W4[i + 1] = V1[i] ^ V2[i - 1];
//     }
//     W0[14] = U1[13];
//     W4[14] = V1[13];
//     for (int32_t i = 0 ; i < 14 ; i++) {
//         W3[i] ^= W0[i];
//         W2[i] ^= W4[i];
//     }
//     W3[14] = W0[14];
//     W2[14] = W4[14];
//     for (int32_t i = 0 ; i < 14 ; i++) {
//         W0[i] ^= U0[i];
//         W4[i] ^= V0[i];
//     }
//     gfmul_15_pclmul(tmp, W3, W2);
//     for (int32_t i = 0 ; i < 30 ; i++) {
//         W3[i] = tmp[i];
//     }
//     gfmul_15_pclmul(W2, W0, W4);
//     gfmul_12_pclmul(W4, U2, V2);
//     gfmul_14_pclmul(W0, U0, V0);
//     for (int32_t i = 0 ; i < 30 ; i++) {
//         W3[i] ^= W2[i];
//     }
//     for (int32_t i = 0 ; i < 28 ; i++) {
//         W1[i] ^= W0[i];
//     }
//     for (int32_t i = 0 ; i < 27 ; i++) {
//         int32_t i1 = i + 1;
//         W2[i] = W2[i1] ^ W0[i1];
//     }
//     W2[27] = W2[28];
//     W2[28] = W2[29];
//     for (int32_t i = 0 ; i < 24 ; i++) {
//         tmp[i] = W2[i] ^ W3[i] ^ W4[i];
//     }
//     for (int32_t i = 24 ; i < 29 ; i++) {
//         tmp[i] = W2[i] ^ W3[i];
//     }
//     tmp[29] = W3[29];
//     for (int32_t i = 0 ; i < 24 ; i++) {
//         tmp[i + 3] ^= W4[i];
//     }
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 30);
//     for (int32_t i = 0 ; i < 27 ; i++) {
//         int32_t i1 = i + 1;
//         tmp[i] = W3[i1] ^ W1[i1];
//     }
//     tmp[27] = W3[28];
//     tmp[28] = W3[29];
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 29);
//     for (int32_t i = 0 ; i < 24 ; i++) {
//         W1[i] ^= W2[i] ^ W4[i];
//     }
//     for (int32_t i = 24 ; i < 28 ; i++) {
//         W1[i] ^= W2[i];
//     }
//     W1[28] = W2[28];
//     for (int32_t i = 0 ; i < 28 ; i++) {
//         W2[i] ^= W3[i];
//     }
//     for (int32_t i = 0; i < 10; i++) {
//         int32_t j = i + 14;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 14] = W1[j] ^ W2[i];
//         Out[j + 28] = W2[j] ^ W3[i];
//         Out[i + 56] = W3[j] ^ W4[i];
//         Out[j + 56] = W4[j];
//     }
//     for (int32_t i = 10; i < 14; i++) {
//         int32_t j = i + 14;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 14] = W1[j] ^ W2[i];
//         Out[j + 28] = W2[j] ^ W3[i];
//         Out[i + 56] = W3[j] ^ W4[i];
//     }
//     Out[42] ^= W1[28];
//     Out[56] ^= W2[28];
// }

//len = 53: TC3_256
static inline void gfmul_53_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[36], W1[37], W2[38], W3[38], W4[34], tmp[38];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[18];
    U2 = (const __m256i *)&A256[36];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[18];
    V2 = (const __m256i *)&B256[36];
    for (int32_t i = 0 ; i < 17 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[17] = U0[17] ^ U1[17];
    W2[17] = V0[17] ^ V1[17];
    gfmul_18_pclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 18 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[18] = W0[18];
    W2[18] = W4[18];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_19_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 38 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_19_pclmul(W2, W0, W4);
    gfmul_17_pclmul(W4, U2, V2);
    gfmul_18_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 38 ; i++) {
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
    for (int32_t i = 0 ; i < 34 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 34 ; i < 37 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[37] = W3[37];
    for (int32_t i = 0 ; i < 34 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 38);
    for (int32_t i = 0 ; i < 35 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[35] = W3[36];
    tmp[36] = W3[37];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 37);
    for (int32_t i = 0 ; i < 34 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 34 ; i < 36 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[36] = W2[36];
    for (int32_t i = 0 ; i < 36 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 16; i++) {
        int32_t j = i + 18;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 18] = W1[j] ^ W2[i];
        Out[j + 36] = W2[j] ^ W3[i];
        Out[i + 72] = W3[j] ^ W4[i];
        Out[j + 72] = W4[j];
    }
    for (int32_t i = 16; i < 18; i++) {
        int32_t j = i + 18;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 18] = W1[j] ^ W2[i];
        Out[j + 36] = W2[j] ^ W3[i];
        Out[i + 72] = W3[j] ^ W4[i];
    }
    Out[54] ^= W1[36];
    Out[72] ^= W2[36];
}

//len = 54: TC3_128
static inline void gfmul_54_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[36], W1[38], W2[38], W3[38], W4[36];
    static __m256i tmp[38];
    __m128i zero128;
    zero128 = _mm_setzero_si128();
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
    gfmul_18_pclmul(W1, W2, W3);
    const __m128i *U1_128 = ((const __m128i *) U1);
    const __m128i *V1_128 = ((const __m128i *) V1);
    W0[0] = _mm256_set_m128i(U1_128[0],zero128);
    W4[0] = _mm256_set_m128i(V1_128[0],zero128);
    U1_128 = ((const __m128i *) U1) + 1;
    V1_128 = ((const __m128i *) V1) + 1;
    for(int32_t i = 0; i < 17; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_set_m128i(V1_128[i2+1],V1_128[i2]);
        W4[i1] ^= V2[i];
    }
    W0[18] = _mm256_set_m128i(zero128,U1_128[34]) ^ U2[17];
    W4[18] = _mm256_set_m128i(zero128,V1_128[34]) ^ V2[17];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[18] = W0[18];
    W2[18] = W4[18];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_19_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 38; i++) {
        W3[i] = tmp[i];
    }
    gfmul_19_pclmul(W2, W0, W4);
    gfmul_18_pclmul(W4, U2, V2);
    gfmul_18_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 38 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 36 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_128 = ((const __m128i *) W2) + 1;
    const __m128i * U2_128 = ((__m128i *) W0) + 1;
    for(int32_t i = 0; i < 35; i++) {
        int32_t i2 = i << 1;
        W2[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W2[i] ^= _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    W2[35]=_mm256_set_m128i(U1_128[71],U2_128[70]^U1_128[70]);
    W2[36]=_mm256_set_m128i(U1_128[73],U1_128[72]);
    W2[37]=_mm256_set_m128i(zero128,U1_128[74]);
    U1_128 = ((__m128i *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ _mm256_set_m128i(U1_128[0],zero128);
    U1_128 = ((const __m128i *) W4) + 1;
    for(int32_t i = 2; i < 36; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^_mm256_set_m128i(U1_128[i2 - 3],U1_128[i2 - 4]);
    }
    tmp[36] = W2[36] ^ W3[36] ^ _mm256_set_m128i(U1_128[69],U1_128[68]);
    tmp[37] = W2[37] ^ W3[37] ^ _mm256_set_m128i(zero128,U1_128[70]);
    divide_by_x_plus_one_128(W2, tmp, 76);
    U1_128 = ((const __m128i *) W3) + 1;
    U2_128 = ((const __m128i *) W1) + 1;
    for(int32_t i = 0; i < 35; i++) {
        int32_t i2 = i << 1;
        tmp[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]) ^ _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    tmp[35]=_mm256_set_m128i(U1_128[71], U2_128[70]^U1_128[70]);
    tmp[36]=_mm256_set_m128i(U1_128[73],U1_128[72]);
    tmp[37]=_mm256_set_m128i(zero128, U1_128[74]);
    divide_by_x_plus_one_128(W3, tmp, 75);
    for (int32_t i = 0 ; i < 36 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[36] = W2[36];
    W1[37] = W2[37];
    for (int32_t i = 0 ; i < 37 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 18; i++)
    {
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
    Out[72] ^= W2[36];
    Out[73] ^= W2[37];
    Out[90] ^= W3[36];
}

// //len = 80: 2-Karatsuba
// static inline void gfmul_80_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[80], D1[80], D2[80], SAA[40], SBB[40];
//     gfmul_40_pclmul(D0, A, B);
//     gfmul_40_pclmul(D2, (A+40), (B+40));
//     for(int32_t i = 0; i < 40; i++) {
//         int32_t is = i + 40;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//     gfmul_40_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 40; i++) {
//         int32_t is = i + 40;
//         int32_t is2 = is + 40;
//         int32_t is3 = is2 + 40;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
// }

//////////////////////////////

//len = 48: 2-Karatsuba
void gfmul_48_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[48], D1[48], D2[48], SAA[24], SBB[24];
    gfmul_24_pclmul(D0, A, B);
    gfmul_24_pclmul(D2, (A+24), (B+24));
    for(int32_t i = 0; i < 24; i++) {
        int32_t is = i + 24;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_24_pclmul(D1, SAA, SBB);
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

// //len = 49: 2-Karatsuba
// void gfmul_49_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[50], D1[50], D2[48], SAA[25], SBB[25];
//     gfmul_25_pclmul(D0, A, B);
//     gfmul_24_pclmul(D2, (A+25), (B+25));
//     for(int32_t i = 0; i < 24; i++) {
//         int32_t is = i + 25;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//         SAA[24]=A[24];        SBB[24]=B[24];    gfmul_25_pclmul(D1, SAA, SBB);
//     for(int32_t i = 0; i < 23; i++) {
//         int32_t is = i + 25;
//         int32_t is2 = is + 25;
//         int32_t is3 = is2 + 25;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is] ^ D2[is];
//         Out[is3] = D2[is];
//     }
//     for(int32_t i = 23; i < 25; i++) {
//         int32_t is = i + 25;
//         int32_t is2 = is + 25;
//         __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
//         Out[i]   = D0[i];
//         Out[is]  = middle ^ D0[i] ^ D1[i];
//         Out[is2] = middle ^ D1[is];
//     }
// }

//len = 49: TC3_128
void gfmul_49_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    static __m256i U0[17], U1[17], U2[16], V0[17], V1[17], V2[16];
    static __m256i W0[34], W1[34], W2[34], W3[35], W4[32];
    static __m256i tmp[35];
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    for(int32_t i = 0; i < 16; i++) {
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 66]));
        V1[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 66]));
        U2[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 132]));
        V2[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 132]));
    }
    U0[16]= (__m256i){A[64], A[65], 0x0ul, 0x0ul};
    V0[16]= (__m256i){B[64], B[65], 0x0ul, 0x0ul};
    U1[16]= (__m256i){A[130], A[131], 0x0ul, 0x0ul};
    V1[16]= (__m256i){B[130], B[131], 0x0ul, 0x0ul};
    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[16] = U0[16] ^ U1[16];
    W2[16] = V0[16] ^ V1[16];
    gfmul_17_pclmul(W1, W2, W3);
    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};
    W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};
    U1_64 = ((const uint64_t *) U1) + 2;
    V1_64 = ((const uint64_t *) V1) + 2;
    for(int32_t i = 0; i < 16; i++) {
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));
        W4[i1] ^= V2[i];
    }
    for (int32_t i = 0 ; i < 17 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0 ; i < 17 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_17_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 34; i++) {
        W3[i] = tmp[i];
    }
    gfmul_17_pclmul(W2, W0, W4);
    gfmul_16_pclmul(W4, U2, V2);
    gfmul_17_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 34 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 33 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_64 = ((const uint64_t *) W2) + 2;
    const uint64_t * U2_64 = ((const uint64_t *) W0) + 2;
    for(int32_t i = 0; i < 32; i++) {
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    W2[32]=(__m256i){U2_64[128], U2_64[129], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[128]));
    W2[33]=(__m256i){U1_64[132], U1_64[133], 0ul, 0ul};
    U1_64 = ((const uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};
    U1_64 = ((const uint64_t *) W4) + 2;
    for(int32_t i = 2; i < 32; i++) {
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));
    }
    tmp[32] = W2[32] ^ W3[32] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[120]));
    tmp[33] = W2[33] ^ W3[33] ^ (__m256i){U1_64[124],U1_64[125], 0ul, 0ul};
    divide_by_x_plus_one_128(W2, tmp, 68);
    U1_64 = ((const uint64_t *) W3) + 2;
    U2_64 = ((const uint64_t *) W1) + 2;
    for(int32_t i = 0; i < 32; i++) {
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    tmp[32]=(__m256i){U2_64[128], U2_64[129], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[128]));
    tmp[33]=(__m256i){U1_64[132], U1_64[133], 0ul, 0ul};
    divide_by_x_plus_one_128(W3, tmp, 67);
    for (int32_t i = 0 ; i < 32 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[32] ^= W2[32];
    W1[33] = W2[33];
    for (int32_t i = 0 ; i < 33 ; i++) {
        W2[i] ^= W3[i];
    }
    for(int32_t i = 0; i < 32; i++) {
        Out[i] = W0[i];
        Out[i + 33] = W2[i];
        Out[i + 66] = W4[i];
    }
    Out[32] = W0[32];
    Out[65] = W2[32];
    Out[66] ^= W2[33];
    uint64_t * Out1, * Out2;
    Out1 = ((uint64_t *) &Out[16]) + 2;
    Out2 = ((uint64_t *) &Out[49]) + 2;
    __m256i aux;
    for(int32_t i = 0; i < 34; i++) {
        int32_t i4 = i << 2;
        aux = _mm256_lddqu_si256 ((__m256i *) (& Out1[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& Out1[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& Out2[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& Out2[i4]), aux);
    }
}

// //len = 96: TC3_256
// void gfmul_96_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
//     const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
//     static __m256i W0[64], W1[67], W2[68], W3[68], W4[64], tmp[68];
//     static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
//     U0 = (const __m256i *)&A256[0];
//     U1 = (const __m256i *)&A256[32];
//     U2 = (const __m256i *)&A256[64];
//     V0 = (const __m256i *)&B256[0];
//     V1 = (const __m256i *)&B256[32];
//     V2 = (const __m256i *)&B256[64];
//     for (int32_t i = 0 ; i < 32 ; i++) {
//         W3[i] = U0[i] ^ U1[i] ^ U2[i];
//         W2[i] = V0[i] ^ V1[i] ^ V2[i];
//     }
//     gfmul_32_pclmul(W1, W2, W3);
//     W0[0] = zero;
//     W4[0] = zero;
//     W0[1] = U1[0];
//     W4[1] = V1[0];
//     for (int32_t i = 1 ; i < 32 ; i++) {
//         W0[i + 1] = U1[i] ^ U2[i - 1];
//         W4[i + 1] = V1[i] ^ V2[i - 1];
//     }
//     W0[33] = U2[31];
//     W4[33] = V2[31];
//     for (int32_t i = 0 ; i < 32 ; i++) {
//         W3[i] ^= W0[i];
//         W2[i] ^= W4[i];
//     }
//     W3[32] = W0[32];
//     W3[33] = W0[33];
//     W2[32] = W4[32];
//     W2[33] = W4[33];
//     for (int32_t i = 0 ; i < 32 ; i++) {
//         W0[i] ^= U0[i];
//         W4[i] ^= V0[i];
//     }
//     gfmul_34_pclmul(tmp, W3, W2);
//     for (int32_t i = 0 ; i < 68 ; i++) {
//         W3[i] = tmp[i];
//     }
//     gfmul_34_pclmul(W2, W0, W4);
//     gfmul_32_pclmul(W4, U2, V2);
//     gfmul_32_pclmul(W0, U0, V0);
//     for (int32_t i = 0 ; i < 68 ; i++) {
//         W3[i] ^= W2[i];
//     }
//     for (int32_t i = 0 ; i < 64 ; i++) {
//         W1[i] ^= W0[i];
//     }
//     for (int32_t i = 0 ; i < 63 ; i++) {
//         int32_t i1 = i + 1;
//         W2[i] = W2[i1] ^ W0[i1];
//     }
//     W2[63] = W2[64];
//     W2[64] = W2[65];
//     W2[65] = W2[66];
//     W2[66] = W2[67];
//     for (int32_t i = 0 ; i < 64 ; i++) {
//         tmp[i] = W2[i] ^ W3[i] ^ W4[i];
//     }
//     for (int32_t i = 64 ; i < 67 ; i++) {
//         tmp[i] = W2[i] ^ W3[i];
//     }
//     tmp[67] = W3[67];
//     for (int32_t i = 0 ; i < 64 ; i++) {
//         tmp[i + 3] ^= W4[i];
//     }
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 68);
//     for (int32_t i = 0 ; i < 63 ; i++) {
//         int32_t i1 = i + 1;
//         tmp[i] = W3[i1] ^ W1[i1];
//     }
//     tmp[63] = W3[64];
//     tmp[64] = W3[65];
//     tmp[65] = W3[66];
//     tmp[66] = W3[67];
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 67);
//     for (int32_t i = 0 ; i < 64 ; i++) {
//         W1[i] ^= W2[i] ^ W4[i];
//     }
//     for (int32_t i = 64 ; i < 67 ; i++) {
//         W1[i] = W2[i];
//     }
//     for (int32_t i = 0 ; i < 66 ; i++) {
//         W2[i] ^= W3[i];
//     }
//     for (int32_t i = 0; i < 32; i++) {
//         int32_t j = i + 32;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 32] = W1[j] ^ W2[i];
//         Out[j + 64] = W2[j] ^ W3[i];
//         Out[i + 128] = W3[j] ^ W4[i];
//         Out[j + 128] = W4[j];
//     }
//     Out[96] ^= W1[64];
//     Out[97] ^= W1[65];
//     Out[98] ^= W1[66];
//     Out[128] ^= W2[64];
//     Out[129] ^= W2[65];
//     Out[130] ^= W2[66];
//     Out[160] ^= W3[64];
//     Out[161] ^= W3[65];
// }

//len = 96: TC3_128
void gfmul_96_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
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
    gfmul_32_pclmul(W1, W2, W3);
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
    gfmul_33_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 66; i++) {
        W3[i] = tmp[i];
    }
    gfmul_33_pclmul(W2, W0, W4);
    gfmul_32_pclmul(W4, U2, V2);
    gfmul_32_pclmul(W0, U0, V0);
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

// //len = 97: TC3_256
// void gfmul_97_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
//     const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
//     static __m256i W0[66], W1[67], W2[68], W3[68], W4[62], tmp[68];
//     static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
//     U0 = (const __m256i *)&A256[0];
//     U1 = (const __m256i *)&A256[33];
//     U2 = (const __m256i *)&A256[66];
//     V0 = (const __m256i *)&B256[0];
//     V1 = (const __m256i *)&B256[33];
//     V2 = (const __m256i *)&B256[66];
//     for (int32_t i = 0 ; i < 31 ; i++) {
//         W3[i] = U0[i] ^ U1[i] ^ U2[i];
//         W2[i] = V0[i] ^ V1[i] ^ V2[i];
//     }
//     W3[31] = U0[31] ^ U1[31];
//     W2[31] = V0[31] ^ V1[31];
//     W3[32] = U0[32] ^ U1[32];
//     W2[32] = V0[32] ^ V1[32];
//     gfmul_33_pclmul(W1, W2, W3);
//     W0[0] = zero;
//     W4[0] = zero;
//     W0[1] = U1[0];
//     W4[1] = V1[0];
//     for (int32_t i = 1 ; i < 32 ; i++) {
//         W0[i + 1] = U1[i] ^ U2[i - 1];
//         W4[i + 1] = V1[i] ^ V2[i - 1];
//     }
//     W0[33] = U1[32];
//     W4[33] = V1[32];
//     for (int32_t i = 0 ; i < 33 ; i++) {
//         W3[i] ^= W0[i];
//         W2[i] ^= W4[i];
//     }
//     W3[33] = W0[33];
//     W2[33] = W4[33];
//     for (int32_t i = 0 ; i < 33 ; i++) {
//         W0[i] ^= U0[i];
//         W4[i] ^= V0[i];
//     }
//     gfmul_34_pclmul(tmp, W3, W2);
//     for (int32_t i = 0 ; i < 68 ; i++) {
//         W3[i] = tmp[i];
//     }
//     gfmul_34_pclmul(W2, W0, W4);
//     gfmul_31_pclmul(W4, U2, V2);
//     gfmul_33_pclmul(W0, U0, V0);
//     for (int32_t i = 0 ; i < 68 ; i++) {
//         W3[i] ^= W2[i];
//     }
//     for (int32_t i = 0 ; i < 66 ; i++) {
//         W1[i] ^= W0[i];
//     }
//     for (int32_t i = 0 ; i < 65 ; i++) {
//         int32_t i1 = i + 1;
//         W2[i] = W2[i1] ^ W0[i1];
//     }
//     W2[65] = W2[66];
//     W2[66] = W2[67];
//     for (int32_t i = 0 ; i < 62 ; i++) {
//         tmp[i] = W2[i] ^ W3[i] ^ W4[i];
//     }
//     for (int32_t i = 62 ; i < 67 ; i++) {
//         tmp[i] = W2[i] ^ W3[i];
//     }
//     tmp[67] = W3[67];
//     for (int32_t i = 0 ; i < 62 ; i++) {
//         tmp[i + 3] ^= W4[i];
//     }
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 68);
//     for (int32_t i = 0 ; i < 65 ; i++) {
//         int32_t i1 = i + 1;
//         tmp[i] = W3[i1] ^ W1[i1];
//     }
//     tmp[65] = W3[66];
//     tmp[66] = W3[67];
//     divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 67);
//     for (int32_t i = 0 ; i < 62 ; i++) {
//         W1[i] ^= W2[i] ^ W4[i];
//     }
//     for (int32_t i = 62 ; i < 66 ; i++) {
//         W1[i] ^= W2[i];
//     }
//     W1[66] = W2[66];
//     for (int32_t i = 0 ; i < 66 ; i++) {
//         W2[i] ^= W3[i];
//     }
//     for (int32_t i = 0; i < 29; i++) {
//         int32_t j = i + 33;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 33] = W1[j] ^ W2[i];
//         Out[j + 66] = W2[j] ^ W3[i];
//         Out[i + 132] = W3[j] ^ W4[i];
//         Out[j + 132] = W4[j];
//     }
//     for (int32_t i = 29; i < 33; i++) {
//         int32_t j = i + 33;
//         Out[i] = W0[i];
//         Out[j] = W0[j] ^ W1[i];
//         Out[j + 33] = W1[j] ^ W2[i];
//         Out[j + 66] = W2[j] ^ W3[i];
//         Out[i + 132] = W3[j] ^ W4[i];
//     }
//     Out[99] ^= W1[66];
//     Out[132] ^= W2[66];
// }

//len = 97: TC3_128
void gfmul_97_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    static __m256i U0[33], U1[33], U2[32], V0[33], V1[33], V2[32];
    static __m256i W0[66], W1[66], W2[66], W3[67], W4[64];
    static __m256i tmp[67];
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    for(int32_t i = 0; i < 32; i++) {
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 130]));
        V1[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 130]));
        U2[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 260]));
        V2[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 260]));
    }
    U0[32]= (__m256i){A[128], A[129], 0x0ul, 0x0ul};
    V0[32]= (__m256i){B[128], B[129], 0x0ul, 0x0ul};
    U1[32]= (__m256i){A[258], A[259], 0x0ul, 0x0ul};
    V1[32]= (__m256i){B[258], B[259], 0x0ul, 0x0ul};
    for (int32_t i = 0 ; i < 32 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[32] = U0[32] ^ U1[32];
    W2[32] = V0[32] ^ V1[32];
    gfmul_33_pclmul(W1, W2, W3);
    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};
    W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};
    U1_64 = ((const uint64_t *) U1) + 2;
    V1_64 = ((const uint64_t *) V1) + 2;
    for(int32_t i = 0; i < 32; i++) {
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));
        W4[i1] ^= V2[i];
    }
    for (int32_t i = 0 ; i < 33 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0 ; i < 33 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_33_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 66; i++) {
        W3[i] = tmp[i];
    }
    gfmul_33_pclmul(W2, W0, W4);
    gfmul_32_pclmul(W4, U2, V2);
    gfmul_33_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 66 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 65 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_64 = ((const uint64_t *) W2) + 2;
    const uint64_t * U2_64 = ((const uint64_t *) W0) + 2;
    for(int32_t i = 0; i < 64; i++) {
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    W2[64]=(__m256i){U2_64[256], U2_64[257], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[256]));
    W2[65]=(__m256i){U1_64[260], U1_64[261], 0ul, 0ul};
    U1_64 = ((const uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};
    U1_64 = ((const uint64_t *) W4) + 2;
    for(int32_t i = 2; i < 64; i++) {
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));
    }
    tmp[64] = W2[64] ^ W3[64] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[248]));
    tmp[65] = W2[65] ^ W3[65] ^ (__m256i){U1_64[252],U1_64[253], 0ul, 0ul};
    divide_by_x_plus_one_128(W2, tmp, 132);
    U1_64 = ((const uint64_t *) W3) + 2;
    U2_64 = ((const uint64_t *) W1) + 2;
    for(int32_t i = 0; i < 64; i++) {
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    tmp[64]=(__m256i){U2_64[256], U2_64[257], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[256]));
    tmp[65]=(__m256i){U1_64[260], U1_64[261], 0ul, 0ul};
    divide_by_x_plus_one_128(W3, tmp, 131);
    for (int32_t i = 0 ; i < 64 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[64] ^= W2[64];
    W1[65] = W2[65];
    for (int32_t i = 0 ; i < 65 ; i++) {
        W2[i] ^= W3[i];
    }
    for(int32_t i = 0; i < 64; i++) {
        Out[i] = W0[i];
        Out[i + 65] = W2[i];
        Out[i + 130] = W4[i];
    }
    Out[64] = W0[64];
    Out[129] = W2[64];
    Out[130] ^= W2[65];
    uint64_t * Out1, * Out2;
    Out1 = ((uint64_t *) &Out[32]) + 2;
    Out2 = ((uint64_t *) &Out[97]) + 2;
    __m256i aux;
    for(int32_t i = 0; i < 66; i++) {
        int32_t i4 = i << 2;
        aux = _mm256_lddqu_si256 ((__m256i *) (& Out1[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& Out1[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& Out2[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& Out2[i4]), aux);
    }
}

// //len = 160: 2-Karatsuba
// void gfmul_160_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
//     static __m256i D0[160], D1[160], D2[160], SAA[80], SBB[80];
//     gfmul_80_pclmul(D0, A, B);
//     gfmul_80_pclmul(D2, (A+80), (B+80));
//     for(int32_t i = 0; i < 80; i++) {
//         int32_t is = i + 80;
//         SAA[i] = A[i] ^ A[is];
//         SBB[i] = B[i] ^ B[is];
//     }
//     gfmul_80_pclmul(D1, SAA, SBB);
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

//len = 160: TC3_64
void gfmul_160_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    static __m256i UV[324];
    static __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[108], W1[108], W2[108], W3[108], W4[108];
    static   __m256i tmp[322];
    static __m256i zero = {0ul, 0ul, 0ul, 0ul};
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    U0 = UV;
    U1 = U0 + 54;
    U2 = U1 + 54;
    V0 = U2 + 54;
    V1 = V0 + 54;
    V2 = V1 + 54;
    for (int32_t i = 0; i < 53; i++){
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4 + 214]));
        V1[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4 + 214]));
        U2[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4 + 428]));
        V2[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4 + 428]));
    }
    U0[53]= (__m256i){A[212], A[213], 0x0ul, 0x0ul};
    V0[53]= (__m256i){B[212], B[213], 0x0ul, 0x0ul};
    U1[53]= (__m256i){A[426], A[427], 0x0ul, 0x0ul};
    V1[53]= (__m256i){B[426], B[427], 0x0ul, 0x0ul};
    U2[53]= zero;
    V2[53]= zero;
    for (int32_t i = 0; i < 53; i++){
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[53] = U0[53] ^ U1[53];
    W2[53] = V0[53] ^ V1[53];
    gfmul_54_pclmul(W1, W2, W3);
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
    for (int32_t i = 0; i < 53; i++){
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));
        W0[i1] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
        W4[i1] = _mm256_lddqu_si256((__m256i   *)(& V1_64[i4]));
        W4[i1] ^= _mm256_lddqu_si256((__m256i   *)(& V2_64[i4]));
    }
    for (int32_t i = 0; i < 54; i++){
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0; i < 54; i++){
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_54_pclmul(tmp, W3, W2);
    for (int32_t i = 0; i < 108; i++){
        W3[i] = tmp[i];
    }
    gfmul_54_pclmul(W2, W0, W4);
    gfmul_53_pclmul(W4, U2, V2);
    gfmul_54_pclmul(W0, U0, V0);
    for (int32_t i = 0; i < 108; i++){
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0; i < 107; i++){
        W1[i] ^= W0[i];
    }
    U1_64 = ((uint64_t *) W2) + 1;
    U2_64 = ((uint64_t *) W0) + 1;
    for (int32_t i = 0; i < 107; i++){
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
    }
    W2[107]=(__m256i){U1_64[428], 0x0ul, 0x0ul, 0x0ul};
    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0] ^ (__m256i){0x0ul, 0x0ul, 0x0ul, U1_64[0]};
    U1_64++;
    for (int32_t i = 1; i < 106; i++){
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i   *)(& U1_64[i4 - 4]));
    }
    tmp[106] = W2[106] ^ W3[106] ^ (__m256i){U1_64[420],U1_64[421],U1_64[422], 0x0ul};
    tmp[107] = W2[107] ^ W3[107];
    W2[107]=zero;
    divide_by_x_plus_one_64(tmp, tmp, 430);
    for (int32_t i = 0; i < 108; i++){
        W2[i] = tmp[i];
    }
    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for (int32_t i = 0; i < 107; i++){
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
    }
    tmp[107]=(__m256i){U1_64[428],0x0ul,0x0ul,0x0ul};
    W3[107]=zero;
    divide_by_x_plus_one_64(tmp, tmp, 429);
    for (int32_t i = 0; i < 108; i++){
        W3[i] = tmp[i];
    }
    for (int32_t i = 0; i < 106; i++){
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[106] ^= W2[106];
    W1[107] = W2[107];
    for (int32_t i = 0; i < 107; i++){
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 106; i++){
        tmp[i] = W0[i];
        tmp[i + 107] = W2[i];
        tmp[i + 214] = W4[i];
    }
        tmp[106] = W0[106];
        tmp[213] = W2[106];
        tmp[214] ^= W2[107];
    U1_64 = ((uint64_t *) &tmp[53]) + 2;
    U2_64 = ((uint64_t *) &tmp[160]) + 2;
    __m256i aux;
    for(int32_t i = 0; i < 107; i++) {
        int32_t i4 = i << 2;
        aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);
    }
    aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[428])) ^ W1[107];
    _mm256_storeu_si256 ((__m256i *) (& U1_64[428]), aux);
    for(int32_t i = 0; i < 320; i++) {
        Out[i] = tmp[i];
    }
}

//len = 161: TC3_64
void gfmul_161_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    static __m256i UV[324];
    static __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[108], W1[108], W2[108], W3[108], W4[108];
    static   __m256i tmp[322];
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    U0 = UV;
    U1 = U0 + 54;
    U2 = U1 + 54;
    V0 = U2 + 54;
    V1 = V0 + 54;
    V2 = V1 + 54;
    for (int32_t i = 0; i < 53; i++){
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4 + 215]));
        V1[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4 + 215]));
        U2[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4 + 430]));
        V2[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4 + 430]));
    }
    U0[53]= (__m256i){A[212], A[213], A[214], 0x0ul};
    V0[53]= (__m256i){B[212], B[213], B[214], 0x0ul};
    U1[53]= (__m256i){A[427], A[428], A[429], 0x0ul};
    V1[53]= (__m256i){B[427], B[428], B[429], 0x0ul};
    U2[53]= (__m256i){A[642], A[643], 0x0ul, 0x0ul};
    V2[53]= (__m256i){B[642], B[643], 0x0ul, 0x0ul};
    for (int32_t i = 0; i < 54; i++){
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_54_pclmul(W1, W2, W3);
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
    for (int32_t i = 0; i < 53; i++){
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));
        W0[i1] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
        W4[i1] = _mm256_lddqu_si256((__m256i   *)(& V1_64[i4]));
        W4[i1] ^= _mm256_lddqu_si256((__m256i   *)(& V2_64[i4]));
    }
    for (int32_t i = 0; i < 54; i++){
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0; i < 54; i++){
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_54_pclmul(tmp, W3, W2);
    for (int32_t i = 0; i < 108; i++){
        W3[i] = tmp[i];
    }
    gfmul_54_pclmul(W2, W0, W4);
    gfmul_54_pclmul(W4, U2, V2);
    gfmul_54_pclmul(W0, U0, V0);
    for (int32_t i = 0; i < 108; i++){
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0; i < 108; i++){
        W1[i] ^= W0[i];
    }
    U1_64 = ((uint64_t *) W2) + 1;
    U2_64 = ((uint64_t *) W0) + 1;
    for (int32_t i = 0; i < 107; i++){
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
    }
    W2[107]=(__m256i){U1_64[428]^U2_64[428], U1_64[429]^U2_64[429], U1_64[430]^U2_64[430], 0x0ul};
    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0] ^ (__m256i){0x0ul, 0x0ul, 0x0ul, U1_64[0]};
    U1_64++;
    for (int32_t i = 1; i < 107; i++){
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i   *)(& U1_64[i4 - 4]));
    }
    tmp[107] = W2[107] ^ W3[107] ^ (__m256i){U1_64[424],U1_64[425],U1_64[426], 0x0ul};
    divide_by_x_plus_one_64(tmp, tmp, 432);
    for (int32_t i = 0; i < 108; i++){
        W2[i] = tmp[i];
    }
    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for (int32_t i = 0; i < 107; i++){
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
    }
    tmp[107]=(__m256i){U1_64[428]^U2_64[428],U1_64[429]^U2_64[429],U1_64[430]^U2_64[430],0x0ul};
    divide_by_x_plus_one_64(tmp, tmp, 431);
    for (int32_t i = 0; i < 108; i++){
        W3[i] = tmp[i];
    }
    for (int32_t i = 0; i < 108; i++){
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 0; i < 108; i++){
        W2[i] ^= W3[i];
    }
        memset((__m256i*)tmp, 0, sizeof(__m256i) * 322);
    for (int32_t i = 0; i < 107; i++){
        tmp[i] = W0[i];
        tmp[i + 215] = W4[i];
    }
        tmp[107] = W0[107];
    U1_64 = ((uint64_t *) &tmp[53]) + 3;
    U2_64 = ((uint64_t *) &tmp[107]) + 2;
    V1_64 = ((uint64_t *) &tmp[161]) + 1;
    for(int32_t i = 0; i < 108; i++) {
        int32_t i4 = i << 2;
        __m256i aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W2[i];
        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& V1_64[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& V1_64[i4]), aux);
    }
    for(int32_t i = 0; i < 322; i++) {
        Out[i] = tmp[i];
    }
}

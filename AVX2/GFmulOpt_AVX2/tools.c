#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gfopt.h"
#ifdef ENV_AVX2
    #ifdef ENV_VPCLMUL
#include "best_alg/best_alg_result_AVX2_VPCLMUL.h"
    #else
#include "best_alg/best_alg_result_AVX2_PCLMUL.h"
    #endif
#else
    #ifdef ENV_NEON
#include "best_alg/best_alg_result_NEON.h"
    #else
#include "best_alg/best_alg_result_new.h"
    #endif
#endif

//#ifdef ENV_AVX2
void fprint_divide_by_x_plus_one_256(FILE * fp){
    fprintf(fp,"static inline void divide_by_x_plus_one_256(__m256i *in, __m256i *out, int32_t size){\n");
    fprintf(fp,"    out[0] = in[0];\n");
    fprintf(fp,"    for(int32_t i = 1; i < size; i++) {\n");
    fprintf(fp,"        out[i] = _mm256_xor_si256(out[i - 1], in[i]);\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n\n");
}

void fprint_divide_by_x_plus_one_128(FILE * fp){
    fprintf(fp,"static inline void divide_by_x_plus_one_128(__m256i *out, __m256i *in, int32_t size){\n");
    fprintf(fp,"    __m128i *A = (__m128i *) in;\n");
    fprintf(fp,"    __m128i *B = (__m128i *) out;\n\n");
    fprintf(fp,"    B[0] = A[0];\n");
    fprintf(fp,"    for(int32_t i = 1; i < size; i++) {\n");
    fprintf(fp,"        B[i] = _mm_xor_si128(B[i - 1], A[i]);\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n\n");
}

void fprint_divide_by_x_plus_one_64(FILE * fp){
    fprintf(fp,"static inline void divide_by_x_plus_one_64(__m256i *out, __m256i *in, int32_t size){\n");
    fprintf(fp,"    uint64_t *A = (uint64_t*) in;\n");
    fprintf(fp,"    uint64_t *B = (uint64_t*) out;\n\n");
    fprintf(fp,"    B[0] = A[0];\n");
    fprintf(fp,"    for(int32_t i = 1; i < size; i++) {\n");
    fprintf(fp,"        B[i]= B[i - 1] ^ A[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    __asm__ volatile(\"\" : : \"r\"(out): \"memory\");\n");
    fprintf(fp,"}\n\n");
}

#ifdef ENV_VPCLMUL
//basemul
void fprint_karat_mult_1_PCLMULQDQ(FILE * fp, char * func_name){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    fprintf(fp,"    __m128i *A128 = (__m128i *)A, *B128 = (__m128i *)B, *Out128 = (__m128i *)Out;\n");
	fprintf(fp,"    __m128i D1[2];\n");
	fprintf(fp,"    __m128i D0[2], D2[2];\n");
	fprintf(fp,"    __m128i Al = _mm_loadu_si128(A128);\n");
	fprintf(fp,"    __m128i Ah = _mm_loadu_si128(A128 + 1);\n");
	fprintf(fp,"    __m128i Bl = _mm_loadu_si128(B128);\n");
	fprintf(fp,"    __m128i Bh = _mm_loadu_si128(B128 + 1);\n\n");

	fprintf(fp,"    //	Compute Al.Bl=D0\n");
	fprintf(fp,"    __m128i DD0 = _mm_clmulepi64_si128(Al, Bl, 0);\n");
	fprintf(fp,"    __m128i DD2 = _mm_clmulepi64_si128(Al, Bl, 0x11);\n");
	fprintf(fp,"    __m128i AAlpAAh = _mm_xor_si128(Al, _mm_shuffle_epi32(Al, 0x4e));\n");
	fprintf(fp,"    __m128i BBlpBBh = _mm_xor_si128(Bl, _mm_shuffle_epi32(Bl, 0x4e)); \n");
	fprintf(fp,"    __m128i DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));\n");
	fprintf(fp,"    D0[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));\n");
	fprintf(fp,"    D0[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));\n\n");

	fprintf(fp,"    //	Compute Ah.Bh=D2\n");
	fprintf(fp,"    DD0 = _mm_clmulepi64_si128(Ah, Bh, 0);\n");
	fprintf(fp,"    DD2 = _mm_clmulepi64_si128(Ah, Bh, 0x11);\n");
	fprintf(fp,"    AAlpAAh = _mm_xor_si128(Ah, _mm_shuffle_epi32(Ah, 0x4e));\n");
	fprintf(fp,"    BBlpBBh = _mm_xor_si128(Bh, _mm_shuffle_epi32(Bh, 0x4e));\n");
	fprintf(fp,"    DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));\n");
	fprintf(fp,"    D2[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));\n");
	fprintf(fp,"    D2[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));\n\n");

	fprintf(fp,"    // Compute AlpAh.BlpBh=D1\n");
	fprintf(fp,"    // Initialisation of AlpAh and BlpBh\n");
	fprintf(fp,"    __m128i AlpAh = _mm_xor_si128(Al,Ah);\n");
	fprintf(fp,"    __m128i BlpBh = _mm_xor_si128(Bl,Bh);\n");
	fprintf(fp,"    DD0 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0);\n");
	fprintf(fp,"    DD2 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0x11);\n");
	fprintf(fp,"    AAlpAAh = _mm_xor_si128(AlpAh, _mm_shuffle_epi32(AlpAh, 0x4e));\n");
	fprintf(fp,"    BBlpBBh = _mm_xor_si128(BlpBh, _mm_shuffle_epi32(BlpBh, 0x4e));\n");
	fprintf(fp,"    DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));\n");
	fprintf(fp,"    D1[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));\n");
	fprintf(fp,"    D1[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));\n\n");

	
	fprintf(fp,"    __m128i middle = _mm_xor_si128(D0[1], D2[0]);\n");
	fprintf(fp,"    Out128[0] = D0[0];\n");
	fprintf(fp,"    Out128[1] = middle ^ D0[0] ^ D1[0];\n");
	fprintf(fp,"    Out128[2] = middle ^ D1[1] ^ D2[1];\n");
	fprintf(fp,"    Out128[3] = D2[1];\n");
	fprintf(fp,"}\n\n");
}

void fprint_karat_karat_1_VPCLMULQDQ(FILE * fp, char * func_name){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    fprintf(fp,"    __m256i T0, T1, T2, T3, T4;\n");
	fprintf(fp,"    __m256i tmp_a, tmp_b, middle, S1, S2, S3;\n");
    fprintf(fp,"    __m256i mask={0,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0};\n\n");

    fprintf(fp,"    T0 = _mm256_clmulepi64_epi128(*A, *B, 0);\n");
    fprintf(fp,"    T1 = _mm256_clmulepi64_epi128(*A, *B, 0x11);\n\n");

    fprintf(fp,"    tmp_a = _mm256_permute4x64_epi64(*A, 0xe1) ^ _mm256_permute4x64_epi64(*A, 0x78);\n");
    fprintf(fp,"    tmp_b = _mm256_permute4x64_epi64(*B, 0xe1) ^ _mm256_permute4x64_epi64(*B, 0x78);\n");
    fprintf(fp,"    T2 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0);\n");
    fprintf(fp,"    T3 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x11);\n\n");

    fprintf(fp,"    tmp_a = tmp_a ^ _mm256_permute4x64_epi64(tmp_a, 0x4e);\n");
    fprintf(fp,"    tmp_b = tmp_b ^ _mm256_permute4x64_epi64(tmp_b, 0x4e);\n");
    fprintf(fp,"    T4 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0);\n\n");

    fprintf(fp,"    // Compute Al.Bl=S1\n");
    fprintf(fp,"    middle = T0 ^ T1 ^ T2;\n");
    fprintf(fp,"    middle = _mm256_permute4x64_epi64(middle, 0xD2);\n");
    fprintf(fp,"    S1 = _mm256_and_si256(middle,mask) ^ _mm256_permute2x128_si256(T0, T1, 0x20);\n\n");

    fprintf(fp,"    // Compute Ah.Bh=S2\n");
    fprintf(fp,"    middle = _mm256_shuffle_epi32(middle, 0x4e);\n");
    fprintf(fp,"    S2 = _mm256_and_si256(middle,mask) ^ _mm256_permute2x128_si256(T0, T1, 0x31);\n\n");

    fprintf(fp,"    // Compute AlpAh.BlpBh=S3\n");
    fprintf(fp,"    S3 = T3 ^ _mm256_and_si256(_mm256_shuffle_epi32(T4, 0x4e),mask);\n");
    fprintf(fp,"    middle = _mm256_permute4x64_epi64(T3, 0xD2);\n");
    fprintf(fp,"    middle = middle ^ _mm256_shuffle_epi32(middle, 0x4e);\n\n");
    
    fprintf(fp,"    //\n");
    fprintf(fp,"    S3 ^= _mm256_and_si256(middle,mask) ^ S1 ^ S2;\n\n");

    fprintf(fp,"    middle = _mm256_setzero_si256();\n");
    fprintf(fp,"    S3=_mm256_permute4x64_epi64(S3, 0x4e);\n");
    fprintf(fp,"    Out[0] = _mm256_xor_si256(S1, _mm256_blend_epi32(S3,middle,0x0F));\n");
    fprintf(fp,"    Out[1] = _mm256_xor_si256(S2, _mm256_blend_epi32(S3,middle,0xF0));\n");
    fprintf(fp,"}\n\n");
}
void fprint_karat_SB_1_VPCLMULQDQ(FILE * fp, char * func_name){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    fprintf(fp,"    __m256i T0, T1, T2;\n");
    fprintf(fp,"    __m256i tmp_a, tmp_b, middle, S1, S2, S3;\n");
    fprintf(fp,"    __m256i mask={0,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0};\n");
    fprintf(fp,"    __m256i mask_rev={0xFFFFFFFFFFFFFFFF,0,0,0xFFFFFFFFFFFFFFFF};\n\n");
 
    fprintf(fp,"    T0 = _mm256_clmulepi64_epi128(*A, *B, 0);\n");
    fprintf(fp,"    T1 = _mm256_clmulepi64_epi128(*A, *B, 0x11);\n\n");

    fprintf(fp,"    tmp_a = *A ^ _mm256_shuffle_epi32(*A, 0x4e);\n");
    fprintf(fp,"    tmp_b = *B ^ _mm256_shuffle_epi32(*B, 0x4e);\n");
    fprintf(fp,"    T2 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0);\n");
    fprintf(fp,"    T2 = T0 ^ T1 ^ T2;\n\n");

    fprintf(fp,"    //S1 = Al*Bl\n");
    fprintf(fp,"    T2=_mm256_permute4x64_epi64(T2, 0xD2);\n");
    fprintf(fp,"    middle = _mm256_and_si256(T2, mask);\n");
    fprintf(fp,"    S1 = _mm256_permute2x128_si256(T0, T1, 0x20) ^ middle;\n\n");

    fprintf(fp,"    //S2 = Ah*Bh\n");
    fprintf(fp,"    T2=_mm256_shuffle_epi32(T2, 0x4e);\n");
    fprintf(fp,"    middle = _mm256_and_si256(T2, mask);\n");
    fprintf(fp,"    S2 = _mm256_permute2x128_si256(T0, T1, 0x31) ^ middle;\n\n");

    fprintf(fp,"    //S3 = AlpAh*BlpBh\n");
    fprintf(fp,"    tmp_a=_mm256_permute4x64_epi64(*A, 0x4e);\n");
    fprintf(fp,"    T0 = _mm256_clmulepi64_epi128(tmp_a, *B, 0);\n");
    fprintf(fp,"    T1 = _mm256_clmulepi64_epi128(tmp_a, *B, 0x11);\n");
    fprintf(fp,"    tmp_a = tmp_a ^ _mm256_shuffle_epi32(tmp_a, 0x4e);\n");
    fprintf(fp,"    T2 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0);\n\n");
    fprintf(fp,"    T2 = T0 ^ T1 ^ T2;\n");
    fprintf(fp,"    T2=_mm256_permute4x64_epi64(T2, 0x2D);\n");
    fprintf(fp,"    middle = _mm256_and_si256(T2, mask_rev);\n");
    fprintf(fp,"    S3 = _mm256_permute2x128_si256(T1, T0, 0x20) ^ middle;\n\n");

    fprintf(fp,"    //\n");
    fprintf(fp,"    T2=_mm256_shuffle_epi32(T2, 0x4e);\n");
    fprintf(fp,"    middle = _mm256_and_si256(T2, mask_rev);\n");
    fprintf(fp,"    S3 ^= _mm256_permute2x128_si256(T1, T0, 0x31) ^ middle;\n\n");
    
    fprintf(fp,"    middle = _mm256_setzero_si256();\n\n");

    fprintf(fp,"    Out[0] = _mm256_xor_si256(S1, _mm256_blend_epi32(S3,middle,0x0F));\n");
    fprintf(fp,"    Out[1] = _mm256_xor_si256(S2, _mm256_blend_epi32(S3,middle,0xF0));\n");
    fprintf(fp,"}\n\n");
}
void fprint_SB_karat_1_VPCLMULQDQ(FILE * fp, char * func_name){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    fprintf(fp,"    __m256i T0, T1, T2;\n");
    fprintf(fp,"    __m256i tmp_a, tmp_b, S0, S1, S2;\n");
    fprintf(fp,"    __m256i zero = _mm256_setzero_si256();\n\n");

    fprintf(fp,"    //S0 = Al*Bl, S1 = Ah*Bh\n");
    fprintf(fp,"    tmp_a = *A;\n");
    fprintf(fp,"    tmp_b = *B;\n");
    fprintf(fp,"    T0 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0);\n");
    fprintf(fp,"    T1 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x01) \n");
    fprintf(fp,"        ^ _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x10);\n");
    fprintf(fp,"    T2 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x11);\n\n");

    fprintf(fp,"    T0 = T0 ^ _mm256_slli_si256(T1,8);\n");
    fprintf(fp,"    T2 = T2 ^ _mm256_srli_si256(T1,8);\n\n");

    fprintf(fp,"    S0 = _mm256_permute2x128_si256(T0, T2, 0x20);\n");
    fprintf(fp,"    S1 = _mm256_permute2x128_si256(T0, T2, 0x31);\n\n");
    
    fprintf(fp,"    //S2 = (Ah + Al) * (Bh + Bl)\n");
    fprintf(fp,"    tmp_a ^= _mm256_permute4x64_epi64(tmp_a, 0x4e);\n");
    fprintf(fp,"    tmp_a = _mm256_permute4x64_epi64(tmp_a, 0xB4);\n\n");

    fprintf(fp,"    tmp_b ^= _mm256_permute4x64_epi64(tmp_b, 0x4e);\n");
    fprintf(fp,"    tmp_b = _mm256_permute4x64_epi64(tmp_b, 0xB4);\n\n");

    fprintf(fp,"    S2 = _mm256_clmulepi64_epi128(tmp_a, tmp_b,0x00);\n");
    fprintf(fp,"    T0 = _mm256_clmulepi64_epi128(tmp_a, tmp_b,0x10);\n\n");

    fprintf(fp,"    S2 ^= _mm256_permute4x64_epi64(T0, 0x99) ^ _mm256_shuffle_epi32(T0, 0x4e); \n\n");

    fprintf(fp,"    //\n");
    fprintf(fp,"    S2 = _mm256_permute4x64_epi64(S0^S1^S2, 0x4e);\n\n");

    fprintf(fp,"    Out[0] = S0 ^ _mm256_blend_epi32(S2,zero,0x0F);\n");
    fprintf(fp,"    Out[1] = S1 ^ _mm256_blend_epi32(S2,zero,0xF0);\n");
    fprintf(fp,"}\n\n");

}
void fprint_SB_SB_1_VPCLMULQDQ(FILE * fp, char * func_name){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);

    fprintf(fp,"    __m256i T0, T1, T2, S0, S1, S2;\n");
    fprintf(fp,"    __m256i tmp_a, tmp_b;\n");
    fprintf(fp,"    __m256i zero = _mm256_setzero_si256();\n\n");

    fprintf(fp,"    //C[0] = Al*Bl, C[1] = Ah*Bh\n");
    fprintf(fp,"    tmp_a = *A;\n");
    fprintf(fp,"    tmp_b = *B;\n");
    fprintf(fp,"    T0 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x00);\n");
    fprintf(fp,"    T1 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x01) \n");
    fprintf(fp,"        ^ _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x10);\n");
    fprintf(fp,"    T2 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x11);\n\n");

    fprintf(fp,"    T0 = T0 ^ _mm256_slli_si256(T1,8);\n");
    fprintf(fp,"    T2 = T2 ^ _mm256_srli_si256(T1,8);\n\n");

    fprintf(fp,"    S0 = _mm256_permute2x128_si256(T0, T2, 0x20);\n");
    fprintf(fp,"    S1 = _mm256_permute2x128_si256(T0, T2, 0x31);\n\n");
    
    fprintf(fp,"    //S2 = Ah*Bl + Al*Bh (128-bit shuffle)\n");
    fprintf(fp,"    tmp_a = _mm256_permute4x64_epi64(tmp_a, 0x4e);\n");
    fprintf(fp,"    T0 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x00);\n");
    fprintf(fp,"    T1 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x01) \n");
    fprintf(fp,"        ^ _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x10);\n");
    fprintf(fp,"    T2 = _mm256_clmulepi64_epi128(tmp_a, tmp_b, 0x11);\n\n");

    fprintf(fp,"    T0 = T0 ^ _mm256_slli_si256(T1,8);\n");
    fprintf(fp,"    T2 = T2 ^ _mm256_srli_si256(T1,8);\n\n");

    fprintf(fp,"    S2 = _mm256_permute2x128_si256(T2, T0, 0x20) ^ _mm256_permute2x128_si256(T2, T0, 0x31);\n\n");

    fprintf(fp,"    //\n");
    fprintf(fp,"    Out[0] = S0 ^ _mm256_blend_epi32(S2,zero,0x0F);\n");
    fprintf(fp,"    Out[1] = S1 ^ _mm256_blend_epi32(S2,zero,0xF0);\n");
    fprintf(fp,"}\n\n");
}


#else
void fprint_karat_mult_1_PCLMULQDQ(FILE * fp, char * func_name){
    fprintf(fp,"static inline void %s(__m128i *Out,   __m128i *A,   __m128i *B){\n",func_name);
	fprintf(fp,"    __m128i D1[2];\n");
	fprintf(fp,"    __m128i D0[2], D2[2];\n");
	fprintf(fp,"    __m128i Al = _mm_loadu_si128(A);\n");
	fprintf(fp,"    __m128i Ah = _mm_loadu_si128(A + 1);\n");
	fprintf(fp,"    __m128i Bl = _mm_loadu_si128(B);\n");
	fprintf(fp,"    __m128i Bh = _mm_loadu_si128(B + 1);\n\n");

	fprintf(fp,"    //	Compute Al.Bl=D0\n");
	fprintf(fp,"    __m128i DD0 = _mm_clmulepi64_si128(Al, Bl, 0);\n");
	fprintf(fp,"    __m128i DD2 = _mm_clmulepi64_si128(Al, Bl, 0x11);\n");
	fprintf(fp,"    __m128i AAlpAAh = _mm_xor_si128(Al, _mm_shuffle_epi32(Al, 0x4e));\n");
	fprintf(fp,"    __m128i BBlpBBh = _mm_xor_si128(Bl, _mm_shuffle_epi32(Bl, 0x4e)); \n");
	fprintf(fp,"    __m128i DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));\n");
	fprintf(fp,"    D0[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));\n");
	fprintf(fp,"    D0[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));\n\n");

	fprintf(fp,"    //	Compute Ah.Bh=D2\n");
	fprintf(fp,"    DD0 = _mm_clmulepi64_si128(Ah, Bh, 0);\n");
	fprintf(fp,"    DD2 = _mm_clmulepi64_si128(Ah, Bh, 0x11);\n");
	fprintf(fp,"    AAlpAAh = _mm_xor_si128(Ah, _mm_shuffle_epi32(Ah, 0x4e));\n");
	fprintf(fp,"    BBlpBBh = _mm_xor_si128(Bh, _mm_shuffle_epi32(Bh, 0x4e));\n");
	fprintf(fp,"    DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));\n");
	fprintf(fp,"    D2[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));\n");
	fprintf(fp,"    D2[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));\n\n");

	fprintf(fp,"    // Compute AlpAh.BlpBh=D1\n");
	fprintf(fp,"    // Initialisation of AlpAh and BlpBh\n");
	fprintf(fp,"    __m128i AlpAh = _mm_xor_si128(Al,Ah);\n");
	fprintf(fp,"    __m128i BlpBh = _mm_xor_si128(Bl,Bh);\n");
	fprintf(fp,"    DD0 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0);\n");
	fprintf(fp,"    DD2 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0x11);\n");
	fprintf(fp,"    AAlpAAh = _mm_xor_si128(AlpAh, _mm_shuffle_epi32(AlpAh, 0x4e));\n");
	fprintf(fp,"    BBlpBBh = _mm_xor_si128(BlpBh, _mm_shuffle_epi32(BlpBh, 0x4e));\n");
	fprintf(fp,"    DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));\n");
	fprintf(fp,"    D1[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));\n");
	fprintf(fp,"    D1[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));\n\n");

	
	fprintf(fp,"    __m128i middle = _mm_xor_si128(D0[1], D2[0]);\n");
	fprintf(fp,"    Out[0] = D0[0];\n");
	fprintf(fp,"    Out[1] = middle ^ D0[0] ^ D1[0];\n");
	fprintf(fp,"    Out[2] = middle ^ D1[1] ^ D2[1];\n");
	fprintf(fp,"    Out[3] = D2[1];\n");
	fprintf(fp,"}\n\n");
}



#endif
//karat, Toom-Cook
void fprint_karat2_2k(FILE * fp, char * func_name, uint64_t len){
    uint64_t k = len / 2;
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    fprintf(fp,"    static __m256i D0[%ld], D1[%ld], D2[%ld], SAA[%ld], SBB[%ld];\n",len,len,len,k,k);
    fprintf(fp,"    gfmul_%ld(D0, A, B);\n",k);
    fprintf(fp,"    gfmul_%ld(D2, (A+%ld), (B+%ld));\n",k,k,k);
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
    fprintf(fp,"        int32_t is = i + %ld;\n",k);
    fprintf(fp,"        SAA[i] = A[i] ^ A[is];\n");
    fprintf(fp,"        SBB[i] = B[i] ^ B[is];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(D1, SAA, SBB);\n",k);
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
    fprintf(fp,"        int32_t is = i + %ld;\n",k);
    fprintf(fp,"        int32_t is2 = is + %ld;\n",k);
    fprintf(fp,"        int32_t is3 = is2 + %ld;\n",k);
    fprintf(fp,"        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);\n");
    fprintf(fp,"        Out[i]   = D0[i];\n");
    fprintf(fp,"        Out[is]  = middle ^ D0[i] ^ D1[i];\n");
    fprintf(fp,"        Out[is2] = middle ^ D1[is] ^ D2[is];\n");
    fprintf(fp,"        Out[is3] = D2[is];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}
void fprint_karat2_2kp1(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    uint64_t k=(len + 1) / 2;
    fprintf(fp,"    static __m256i D0[%ld], D1[%ld], D2[%ld], SAA[%ld], SBB[%ld];\n",len+1,len+1,len-1,k,k);
    fprintf(fp,"    gfmul_%ld(D0, A, B);\n",k);
    fprintf(fp,"    gfmul_%ld(D2, (A+%ld), (B+%ld));\n",k-1,k,k);
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k-1);
    fprintf(fp,"        int32_t is = i + %ld;\n",k);
    fprintf(fp,"        SAA[i] = A[i] ^ A[is];\n");
    fprintf(fp,"        SBB[i] = B[i] ^ B[is];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"        SAA[%ld]=A[%ld];",k-1,k-1);
    fprintf(fp,"        SBB[%ld]=B[%ld];",k-1,k-1);
    fprintf(fp,"    gfmul_%ld(D1, SAA, SBB);\n",k);
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k-2);
    fprintf(fp,"        int32_t is = i + %ld;\n",k);
    fprintf(fp,"        int32_t is2 = is + %ld;\n",k);
    fprintf(fp,"        int32_t is3 = is2 + %ld;\n",k);
    fprintf(fp,"        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);\n");
    fprintf(fp,"        Out[i]   = D0[i];\n");
    fprintf(fp,"        Out[is]  = middle ^ D0[i] ^ D1[i];\n");
    fprintf(fp,"        Out[is2] = middle ^ D1[is] ^ D2[is];\n");
    fprintf(fp,"        Out[is3] = D2[is];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for(int32_t i = %ld; i < %ld; i++) {\n",k-2,k);
    fprintf(fp,"        int32_t is = i + %ld;\n",k);
    fprintf(fp,"        int32_t is2 = is + %ld;\n",k);
    fprintf(fp,"        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);\n");
    fprintf(fp,"        Out[i]   = D0[i];\n");
    fprintf(fp,"        Out[is]  = middle ^ D0[i] ^ D1[i];\n");
    fprintf(fp,"        Out[is2] = middle ^ D1[is];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}

void fprint_karat3_3k(FILE * fp, char * func_name, uint64_t len){
    uint64_t k=len/3;
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    fprintf(fp,"    __m256i *a0, *a1, *a2, *b0, *b1, *b2, middle;\n");
    fprintf(fp,"    static __m256i aa01[%ld], bb01[%ld], aa02[%ld], bb02[%ld], aa12[%ld], bb12[%ld];\n",k,k,k,k,k,k);
    fprintf(fp,"    static __m256i D0[%ld], D1[%ld], D2[%ld], D3[%ld], D4[%ld], D5[%ld];\n",2*k,2*k,2*k,2*k,2*k,2*k);
    fprintf(fp,"    a0 = A;\n");
    fprintf(fp,"    a1 = A + %ld;\n",k);
    fprintf(fp,"    a2 = A + %ld;\n",2*k);
    fprintf(fp,"    b0 = B;\n");
    fprintf(fp,"    b1 = B + %ld;\n",k);
    fprintf(fp,"    b2 = B + %ld;\n",2*k);
    fprintf(fp,"    for (int16_t i = 0; i < %ld; i++)\n",k);
    fprintf(fp,"    {\n");
    fprintf(fp,"        aa01[i] = a0[i] ^ a1[i];\n");
    fprintf(fp,"        bb01[i] = b0[i] ^ b1[i];\n");
    fprintf(fp,"        aa12[i] = a2[i] ^ a1[i];\n");
    fprintf(fp,"        bb12[i] = b2[i] ^ b1[i];\n");
    fprintf(fp,"        aa02[i] = a0[i] ^ a2[i];\n");
    fprintf(fp,"        bb02[i] = b0[i] ^ b2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(D3, aa01, bb01);\n",k);
    fprintf(fp,"    gfmul_%ld(D4, aa02, bb02);\n",k);
    fprintf(fp,"    gfmul_%ld(D5, aa12, bb12);\n",k);
    fprintf(fp,"    gfmul_%ld(D0, a0, b0);\n",k);
    fprintf(fp,"    gfmul_%ld(D1, a1, b1);\n",k);
    fprintf(fp,"    gfmul_%ld(D2, a2, b2);\n",k);
    fprintf(fp,"    for (int16_t i = 0; i < %ld; i++)\n",k);
    fprintf(fp,"    {\n");
    fprintf(fp,"        int16_t j = i + %ld;\n",k);
    fprintf(fp,"        middle = D0[i] ^ D1[i] ^ D0[j];\n");
    fprintf(fp,"        Out[i] = D0[i];\n");
    fprintf(fp,"        Out[j] = D3[i] ^ middle;\n");
    fprintf(fp,"        Out[j + %ld] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;\n",k);
    fprintf(fp,"        middle = D1[j] ^ D2[i] ^ D2[j];\n");
    fprintf(fp,"        Out[j + %ld] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;\n",k*2);
    fprintf(fp,"        Out[i + %ld] = D5[j] ^ middle;\n",k*4);
    fprintf(fp,"        Out[j + %ld] = D2[j];\n",k*4);
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}
void fprint_karat3_3kp1(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    uint64_t k=len/3;
    fprintf(fp,"    __m256i *a0, *a1, *a2, *b0, *b1, *b2, middle;\n");
    fprintf(fp,"    static __m256i aa01[%ld], bb01[%ld], aa02[%ld], bb02[%ld], aa12[%ld], bb12[%ld];\n",k,k,k+1,k+1,k+1,k+1);
    fprintf(fp,"    static __m256i D0[%ld], D1[%ld], D2[%ld], D3[%ld], D4[%ld], D5[%ld];\n",2*k,2*k,2*k+2,2*k,2*k+2,2*k+2);
    fprintf(fp,"    a0 = A;\n");
    fprintf(fp,"    a1 = A + %ld;\n",k);
    fprintf(fp,"    a2 = A + %ld;\n",2*k);
    fprintf(fp,"    b0 = B;\n");
    fprintf(fp,"    b1 = B + %ld;\n",k);
    fprintf(fp,"    b2 = B + %ld;\n",2*k);
    fprintf(fp,"    for (int16_t i = 0; i < %ld; i++)\n",k);
    fprintf(fp,"    {\n");
    fprintf(fp,"        aa01[i] = a0[i] ^ a1[i];\n");
    fprintf(fp,"        bb01[i] = b0[i] ^ b1[i];\n");
    fprintf(fp,"        aa12[i] = a2[i] ^ a1[i];\n");
    fprintf(fp,"        bb12[i] = b2[i] ^ b1[i];\n");
    fprintf(fp,"        aa02[i] = a0[i] ^ a2[i];\n");
    fprintf(fp,"        bb02[i] = b0[i] ^ b2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    aa12[%ld] = a2[%ld];\n",k,k);
    fprintf(fp,"    bb12[%ld] = b2[%ld];\n",k,k);
    fprintf(fp,"    aa02[%ld] = a2[%ld];\n",k,k);
    fprintf(fp,"    bb02[%ld] = b2[%ld];\n",k,k);
    fprintf(fp,"    gfmul_%ld(D3, aa01, bb01);\n",k);
    fprintf(fp,"    gfmul_%ld(D4, aa02, bb02);\n",k+1);
    fprintf(fp,"    gfmul_%ld(D5, aa12, bb12);\n",k+1);
    fprintf(fp,"    gfmul_%ld(D0, a0, b0);\n",k);
    fprintf(fp,"    gfmul_%ld(D1, a1, b1);\n",k);
    fprintf(fp,"    gfmul_%ld(D2, a2, b2);\n",k+1);
    fprintf(fp,"    for (int16_t i = 0; i < %ld; i++)\n",k);
    fprintf(fp,"    {\n");
    fprintf(fp,"        int16_t j = i + %ld;\n",k);
    fprintf(fp,"        middle = D0[i] ^ D1[i] ^ D0[j];\n");
    fprintf(fp,"        Out[i] = D0[i];\n");
    fprintf(fp,"        Out[j] = D3[i] ^ middle;\n");
    fprintf(fp,"        Out[j + %ld] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;\n",k);
    fprintf(fp,"        middle = D1[j] ^ D2[i] ^ D2[j];\n");
    fprintf(fp,"        Out[j + %ld] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;\n",k*2);
    fprintf(fp,"        Out[i + %ld] = D5[j] ^ middle;\n",k*4);
    fprintf(fp,"        Out[j + %ld] = D2[j];\n",k*4);
    fprintf(fp,"    }\n");
    fprintf(fp,"    Out[%ld] ^= D2[%ld] ^ D4[%ld];\n",4*k,2*k,2*k);
    fprintf(fp,"    Out[%ld] ^= D2[%ld] ^ D4[%ld];\n",4*k+1,2*k+1,2*k+1);
    fprintf(fp,"    Out[%ld] ^= D2[%ld] ^ D5[%ld];\n",5*k,2*k,2*k);
    fprintf(fp,"    Out[%ld] ^= D2[%ld] ^ D5[%ld];\n",5*k+1,2*k+1,2*k+1);
    fprintf(fp,"    Out[%ld] = D2[%ld];\n",6*k,2*k);
    fprintf(fp,"    Out[%ld] = D2[%ld];\n",6*k+1,2*k+1);
    fprintf(fp,"}\n");
    fprintf(fp,"\n");

}
void fprint_karat3_3kp2(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    uint64_t k=(len+2)/3;
    fprintf(fp,"    __m256i *a0, *a1, *a2, *b0, *b1, *b2, middle;\n");
    fprintf(fp,"    static __m256i aa01[%ld], bb01[%ld], aa02[%ld], bb02[%ld], aa12[%ld], bb12[%ld];\n",k,k,k,k,k,k);
    fprintf(fp,"    static __m256i D0[%ld], D1[%ld], D2[%ld], D3[%ld], D4[%ld], D5[%ld];\n",2*k,2*k,2*k,2*k,2*k,2*k);
    fprintf(fp,"    a0 = A;\n");
    fprintf(fp,"    a1 = A + %ld;\n",k);
    fprintf(fp,"    a2 = A + %ld;\n",2*k);
    fprintf(fp,"    b0 = B;\n");
    fprintf(fp,"    b1 = B + %ld;\n",k);
    fprintf(fp,"    b2 = B + %ld;\n",2*k);
    fprintf(fp,"    for (int16_t i = 0; i < %ld; i++)\n",k-1);
    fprintf(fp,"    {\n");
    fprintf(fp,"        aa01[i] = a0[i] ^ a1[i];\n");
    fprintf(fp,"        bb01[i] = b0[i] ^ b1[i];\n");
    fprintf(fp,"        aa12[i] = a2[i] ^ a1[i];\n");
    fprintf(fp,"        bb12[i] = b2[i] ^ b1[i];\n");
    fprintf(fp,"        aa02[i] = a0[i] ^ a2[i];\n");
    fprintf(fp,"        bb02[i] = b0[i] ^ b2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    aa01[%ld] = a0[%ld] ^ a1[%ld];\n",k-1,k-1,k-1);
    fprintf(fp,"    bb01[%ld] = b0[%ld] ^ b1[%ld];\n",k-1,k-1,k-1);
    fprintf(fp,"    aa12[%ld] = a1[%ld];\n",k-1,k-1);
    fprintf(fp,"    bb12[%ld] = b1[%ld];\n",k-1,k-1);
    fprintf(fp,"    aa02[%ld] = a0[%ld];\n",k-1,k-1);
    fprintf(fp,"    bb02[%ld] = b0[%ld];\n",k-1,k-1);
    fprintf(fp,"    gfmul_%ld(D3, aa01, bb01);\n",k);
    fprintf(fp,"    gfmul_%ld(D4, aa02, bb02);\n",k);
    fprintf(fp,"    gfmul_%ld(D5, aa12, bb12);\n",k);
    fprintf(fp,"    gfmul_%ld(D0, a0, b0);\n",k);
    fprintf(fp,"    gfmul_%ld(D1, a1, b1);\n",k);
    fprintf(fp,"    gfmul_%ld(D2, a2, b2);\n",k-1);
    fprintf(fp,"    for (int16_t i = 0; i < %ld; i++)\n",k-2);
    fprintf(fp,"    {\n");
    fprintf(fp,"        int16_t j = i + %ld;\n",k);
    fprintf(fp,"        middle = D0[i] ^ D1[i] ^ D0[j];\n");
    fprintf(fp,"        Out[i] = D0[i];\n");
    fprintf(fp,"        Out[j] = D3[i] ^ middle;\n");
    fprintf(fp,"        Out[j + %ld] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;\n",k);
    fprintf(fp,"        middle = D1[j] ^ D2[i] ^ D2[j];\n");
    fprintf(fp,"        Out[j + %ld] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;\n",2*k);
    fprintf(fp,"        Out[i + %ld] = D5[j] ^ middle;\n",4*k);
    fprintf(fp,"        Out[j + %ld] = D2[j];\n",4*k);
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int16_t i = %ld; i < %ld; i++)\n",k-2,k);
    fprintf(fp,"    {\n");
    fprintf(fp,"        int16_t j = i + %ld;\n",k);
    fprintf(fp,"        middle = D0[i] ^ D1[i] ^ D0[j];\n");
    fprintf(fp,"        Out[i] = D0[i];\n");
    fprintf(fp,"        Out[j] = D3[i] ^ middle;\n");
    fprintf(fp,"        Out[j + %ld] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;\n",k);
    fprintf(fp,"        middle = D1[j] ^ D2[i];\n");
    fprintf(fp,"        Out[j + %ld] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;\n",k*2);
    fprintf(fp,"        Out[i + %ld] = D5[j] ^ middle;\n",4*k);
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}

void fprint_karat5_5k(FILE * fp, char * func_name, uint64_t len){
    uint64_t k=len/5;
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A,   __m256i *B){\n",func_name);
    fprintf(fp,"         __m256i *a0, *b0, *a1, *b1, *a2, *b2, * a3, * b3, *a4, *b4;\n");
    fprintf(fp,"    static __m256i aa01[%ld], bb01[%ld], aa02[%ld], bb02[%ld], aa03[%ld], bb03[%ld],\n",k,k,k,k,k,k);
    fprintf(fp,"               aa04[%ld], bb04[%ld], aa12[%ld], bb12[%ld], aa13[%ld], bb13[%ld],\n",k,k,k,k,k,k);
    fprintf(fp,"               aa14[%ld], bb14[%ld], aa23[%ld], bb23[%ld], aa24[%ld], bb24[%ld], aa34[%ld], bb34[%ld];\n",k,k,k,k,k,k,k,k);
    fprintf(fp,"    static __m256i D0[%ld], D1[%ld], D2[%ld], D3[%ld], D4[%ld],\n",2*k,2*k,2*k,2*k,2*k);
    fprintf(fp,"               D01[%ld], D02[%ld], D03[%ld], D04[%ld], D12[%ld],\n",2*k,2*k,2*k,2*k,2*k);
    fprintf(fp,"               D13[%ld], D14[%ld], D23[%ld], D24[%ld], D34[%ld];\n",2*k,2*k,2*k,2*k,2*k);
    
    fprintf(fp,"    a0 = A;\n");
    fprintf(fp,"    a1 = a0 + %ld;\n",k);
    fprintf(fp,"    a2 = a1 + %ld;\n",k);
    fprintf(fp,"    a3 = a2 + %ld;\n",k);
    fprintf(fp,"    a4 = a3 + %ld;\n",k);
    fprintf(fp,"    b0 = B;\n");
    fprintf(fp,"    b1 = b0 + %ld;\n",k);
    fprintf(fp,"    b2 = b1 + %ld;\n",k);
    fprintf(fp,"    b3 = b2 + %ld;\n",k);
    fprintf(fp,"    b4 = b3 + %ld;\n",k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        aa01[i] = a0[i] ^ a1[i];\n");
    fprintf(fp,"        bb01[i] = b0[i] ^ b1[i];\n");
    fprintf(fp,"        aa02[i] = a0[i] ^ a2[i];\n");
    fprintf(fp,"        bb02[i] = b0[i] ^ b2[i];\n");
    fprintf(fp,"        aa03[i] = a0[i] ^ a3[i];\n");
    fprintf(fp,"        bb03[i] = b0[i] ^ b3[i];\n");
    fprintf(fp,"        aa04[i] = a0[i] ^ a4[i];\n");
    fprintf(fp,"        bb04[i] = b0[i] ^ b4[i];\n");
    fprintf(fp,"        aa12[i] = a1[i] ^ a2[i];\n");
    fprintf(fp,"        bb12[i] = b1[i] ^ b2[i];\n");
    fprintf(fp,"        aa13[i] = a1[i] ^ a3[i];\n");
    fprintf(fp,"        bb13[i] = b1[i] ^ b3[i];\n");
    fprintf(fp,"        aa14[i] = a1[i] ^ a4[i];\n");
    fprintf(fp,"        bb14[i] = b1[i] ^ b4[i];\n");
    fprintf(fp,"        aa23[i] = a2[i] ^ a3[i];\n");
    fprintf(fp,"        bb23[i] = b2[i] ^ b3[i];\n");
    fprintf(fp,"        aa24[i] = a2[i] ^ a4[i];\n");
    fprintf(fp,"        bb24[i] = b2[i] ^ b4[i];\n");
    fprintf(fp,"        aa34[i] = a3[i] ^ a4[i];\n");
    fprintf(fp,"        bb34[i] = b3[i] ^ b4[i];\n");
    fprintf(fp,"    }\n");
    
    fprintf(fp,"    gfmul_%ld(D01, aa01, bb01);\n",k);
    fprintf(fp,"    gfmul_%ld(D02, aa02, bb02);\n",k);
    fprintf(fp,"    gfmul_%ld(D03, aa03, bb03);\n",k);
    fprintf(fp,"    gfmul_%ld(D04, aa04, bb04);\n",k);
    
    fprintf(fp,"    gfmul_%ld(D12, aa12, bb12);\n",k);
    fprintf(fp,"    gfmul_%ld(D13, aa13, bb13);\n",k);
    fprintf(fp,"    gfmul_%ld(D14, aa14, bb14);\n",k);
    
    fprintf(fp,"    gfmul_%ld(D23, aa23, bb23);\n",k);
    fprintf(fp,"    gfmul_%ld(D24, aa24, bb24);\n",k);
    
    fprintf(fp,"    gfmul_%ld(D34, aa34, bb34);\n",k);

    fprintf(fp,"    gfmul_%ld(D0, a0, b0);\n",k);
    fprintf(fp,"    gfmul_%ld(D1, a1, b1);\n",k);
    fprintf(fp,"    gfmul_%ld(D2, a2, b2);\n",k);
    fprintf(fp,"    gfmul_%ld(D3, a3, b3);\n",k);
    fprintf(fp,"    gfmul_%ld(D4, a4, b4);\n",k);
    
    fprintf(fp,"    for (int16_t i = 0 ; i < %ld ; i++) {\n",k);
    
    fprintf(fp,"        int16_t j = i + %ld;\n",k);
    fprintf(fp,"        Out[i] = D0[i];\n");
    fprintf(fp,"        Out[i + %ld] = D0[j] ^ D01[i] ^ D0[i] ^ D1[i];\n",k);
    fprintf(fp,"        Out[i + %ld] = D1[i] ^ D02[i] ^ D0[i] ^ D2[i] ^ D01[j] ^ D0[j] ^ D1[j];\n",2*k);
    fprintf(fp,"        Out[i + %ld] = D1[j] ^ D03[i] ^ D0[i] ^ D3[i] ^ D12[i] ^ D1[i] ^ D2[i] ^ D02[j] ^ D0[j] ^ D2[j];\n",3*k);
    fprintf(fp,"        Out[i + %ld] = D2[i] ^ D04[i] ^ D0[i] ^ D4[i] ^ D13[i] ^ D1[i] ^ D3[i] ^ D03[j] ^ D0[j] ^ D3[j] ^ D12[j] ^ D1[j] ^ D2[j];\n",4*k);
    fprintf(fp,"        Out[i + %ld] = D2[j] ^ D14[i] ^ D1[i] ^ D4[i] ^ D23[i] ^ D2[i] ^ D3[i] ^ D04[j] ^ D0[j] ^ D4[j] ^ D13[j] ^ D1[j] ^ D3[j];\n",5*k);
    fprintf(fp,"        Out[i + %ld] = D3[i] ^ D24[i] ^ D2[i] ^ D4[i] ^ D14[j] ^ D1[j] ^ D4[j] ^ D23[j] ^ D2[j] ^ D3[j];\n",6*k);
    fprintf(fp,"        Out[i + %ld] = D3[j] ^ D34[i] ^ D3[i] ^ D4[i] ^ D24[j] ^ D2[j] ^ D4[j];\n",7*k);
    fprintf(fp,"        Out[i + %ld] = D4[i] ^ D34[j] ^ D3[j] ^ D4[j];\n",8*k);
    fprintf(fp,"        Out[i + %ld] = D4[j];\n",9*k);
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}

void fprint_TC3_256_3k(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A256,   __m256i *B256){\n",func_name);
    uint64_t k=len/3;
    fprintf(fp,"    __m256i *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp,"        static __m256i W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld], tmp[%ld];\n",2*k,2*k+3,2*k+4,2*k+4,2*k,2*k+4);
    fprintf(fp,"    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};\n");
    fprintf(fp,"    U0 = (__m256i *)&A256[0];\n");
    fprintf(fp,"    U1 = (__m256i *)&A256[%ld];\n",k);
    fprintf(fp,"    U2 = (__m256i *)&A256[%ld];\n",2*k);
    fprintf(fp,"    V0 = (__m256i *)&B256[0];\n");
    fprintf(fp,"    V1 = (__m256i *)&B256[%ld];\n",k);
    fprintf(fp,"    V2 = (__m256i *)&B256[%ld];\n",2*k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k);
    fprintf(fp,"    W0[0] = zero;\n");
    fprintf(fp,"    W4[0] = zero;\n");
    fprintf(fp,"    W0[1] = U1[0];\n");
    fprintf(fp,"    W4[1] = V1[0];\n");
    fprintf(fp,"    for (int32_t i = 1 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W0[i + 1] = U1[i] ^ U2[i - 1];\n");
    fprintf(fp,"        W4[i + 1] = V1[i] ^ V2[i - 1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W0[%ld] = U2[%ld];\n",k+1,k-1);
    fprintf(fp,"    W4[%ld] = V2[%ld];\n",k+1,k-1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = W0[%ld];\n",k,k);
    fprintf(fp,"    W3[%ld] = W0[%ld];\n",k+1,k+1);
    fprintf(fp,"    W2[%ld] = W4[%ld];\n",k,k);
    fprintf(fp,"    W2[%ld] = W4[%ld];\n",k+1,k+1);        
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+2);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+4);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+2);
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+4);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W2[i] = W2[i1] ^ W0[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k-1,2*k);
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k,2*k+1);
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k+1,2*k+2);
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k+2,2*k+3);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k,2*k+3);
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+3,2*k+3);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        tmp[i + 3] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, %ld);\n",2*k+4);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        tmp[i] = W3[i1] ^ W1[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k-1,2*k);
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k,2*k+1);
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+1,2*k+2);
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+2,2*k+3);
    fprintf(fp,"    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, %ld);\n",2*k+3);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k,2*k+3);
    fprintf(fp,"        W1[i] = W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++) {\n",k);
    fprintf(fp,"        int32_t j = i + %ld;\n",k);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",k);
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k);
    fprintf(fp,"        Out[j + %ld] = W4[j];\n",4*k);
    fprintf(fp,"    }\n");
    fprintf(fp,"    Out[%ld] ^= W1[%ld];\n",3*k,2*k);
    fprintf(fp,"    Out[%ld] ^= W1[%ld];\n",3*k+1,2*k+1);
    fprintf(fp,"    Out[%ld] ^= W1[%ld];\n",3*k+2,2*k+2);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",4*k,2*k);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",4*k+1,2*k+1);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",4*k+2,2*k+2);
    fprintf(fp,"    Out[%ld] ^= W3[%ld];\n",5*k,2*k);
    fprintf(fp,"    Out[%ld] ^= W3[%ld];\n",5*k+1,2*k+1);
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}
void fprint_TC3_256_3kp1(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A256,   __m256i *B256){\n",func_name);

    uint64_t k=(len+2)/3;
    fprintf(fp,"    __m256i *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp,"        static __m256i W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld], tmp[%ld];\n",2*k,2*k+1,2*k+2,2*k+2,2*k-4,2*k+2);
    fprintf(fp,"    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};\n");
    fprintf(fp,"    U0 = (__m256i *)&A256[0];\n");
    fprintf(fp,"    U1 = (__m256i *)&A256[%ld];\n",k);
    fprintf(fp,"    U2 = (__m256i *)&A256[%ld];\n",2*k);
    fprintf(fp,"    V0 = (__m256i *)&B256[0];\n");
    fprintf(fp,"    V1 = (__m256i *)&B256[%ld];\n",k);
    fprintf(fp,"    V2 = (__m256i *)&B256[%ld];\n",2*k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k-2);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k-2,k-2,k-2);
    fprintf(fp,"    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k-2,k-2,k-2);
    fprintf(fp,"    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k-1,k-1,k-1);
    fprintf(fp,"    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k-1,k-1,k-1);
    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k);
    fprintf(fp,"    W0[0] = zero;\n");
    fprintf(fp,"    W4[0] = zero;\n");
    fprintf(fp,"    W0[1] = U1[0];\n");
    fprintf(fp,"    W4[1] = V1[0];\n");
    fprintf(fp,"    for (int32_t i = 1 ; i < %ld ; i++) {\n",k-1);
    fprintf(fp,"        W0[i + 1] = U1[i] ^ U2[i - 1];\n");
    fprintf(fp,"        W4[i + 1] = V1[i] ^ V2[i - 1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    \n");
    fprintf(fp,"    W0[%ld] = U1[%ld];\n",k,k-1);
    fprintf(fp,"    W4[%ld] = V1[%ld];\n",k,k-1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = W0[%ld];\n",k,k);
    fprintf(fp,"    W2[%ld] = W4[%ld];\n",k,k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+1);
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k-2);
    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W2[i] = W2[i1] ^ W0[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k-1,2*k);
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k,2*k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-4);
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k-4,2*k+1);
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+1,2*k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-4);
    fprintf(fp,"        tmp[i + 3] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, %ld);\n",2*k+2);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        tmp[i] = W3[i1] ^ W1[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k-1,2*k);
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k,2*k+1);
    fprintf(fp,"    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, %ld);\n",2*k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-4);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k-4,2*k);
    fprintf(fp,"        W1[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W1[%ld] = W2[%ld];\n",2*k,2*k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++) {\n",k-4);
    fprintf(fp,"        int32_t j = i + %ld;\n",k);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",k);
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k);
    fprintf(fp,"        Out[j + %ld] = W4[j];\n",4*k);
    fprintf(fp,"    }\n");        
    fprintf(fp,"    for (int32_t i = %ld; i < %ld; i++) {\n",k-4,k);
    fprintf(fp,"        int32_t j = i + %ld;\n",k);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",k);
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k);
    fprintf(fp,"    }\n");
    fprintf(fp,"    Out[%ld] ^= W1[%ld];\n",3*k,2*k);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",4*k,2*k);
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}
void fprint_TC3_256_3kp2(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A256,   __m256i *B256){\n",func_name);
    uint64_t k=(len+2)/3;
    fprintf(fp,"    __m256i *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp,"        static __m256i W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld], tmp[%ld];\n",2*k,2*k+1,2*k+2,2*k+2,2*k-2,2*k+2);
    fprintf(fp,"    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};\n");
    fprintf(fp,"    U0 = (__m256i *)&A256[0];\n");
    fprintf(fp,"    U1 = (__m256i *)&A256[%ld];\n",k);
    fprintf(fp,"    U2 = (__m256i *)&A256[%ld];\n",2*k);
    fprintf(fp,"    V0 = (__m256i *)&B256[0];\n");
    fprintf(fp,"    V1 = (__m256i *)&B256[%ld];\n",k);
    fprintf(fp,"    V2 = (__m256i *)&B256[%ld];\n",2*k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k-1);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k-1,k-1,k-1);
    fprintf(fp,"    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k-1,k-1,k-1);
    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k);
    fprintf(fp,"    W0[0] = zero;\n");
    fprintf(fp,"    W4[0] = zero;\n");
    fprintf(fp,"    W0[1] = U1[0];\n");
    fprintf(fp,"    W4[1] = V1[0];\n");
    fprintf(fp,"    for (int32_t i = 1 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W0[i + 1] = U1[i] ^ U2[i - 1];\n");
    fprintf(fp,"        W4[i + 1] = V1[i] ^ V2[i - 1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = W0[%ld];\n",k,k);
    fprintf(fp,"    W2[%ld] = W4[%ld];\n",k,k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+1);
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k-1);
    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W2[i] = W2[i1] ^ W0[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k-1,2*k);
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k,2*k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-2);
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k-2,2*k+1);
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+1,2*k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-2);
    fprintf(fp,"        tmp[i + 3] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, %ld);\n",2*k+2);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        tmp[i] = W3[i1] ^ W1[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k-1,2*k);
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k,2*k+1);
    fprintf(fp,"    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, %ld);\n",2*k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-2);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k-2,2*k);
    fprintf(fp,"        W1[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W1[%ld] = W2[%ld];\n",2*k,2*k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++) {\n",k-2);
    fprintf(fp,"        int32_t j = i + %ld;\n",k);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",k);
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k);
    fprintf(fp,"        Out[j + %ld] = W4[j];\n",4*k);
    fprintf(fp,"    }\n");        
    fprintf(fp,"    for (int32_t i = %ld; i < %ld; i++) {\n",k-2,k);
    fprintf(fp,"        int32_t j = i + %ld;\n",k);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",k);
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k);
    fprintf(fp,"    }\n");
    fprintf(fp,"    Out[%ld] ^= W1[%ld];\n",3*k,2*k);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",4*k,2*k);
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}

void fprint_TC3_128_3k(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A256,   __m256i *B256){\n",func_name);
    
    uint64_t k=len/3;

    fprintf(fp,"    __m256i *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp,"    static __m256i W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld];\n",k*2,k*2+2,k*2+2,k*2+2,k*2);
    fprintf(fp,"    static __m256i tmp[%ld];\n",k*2+2);

    fprintf(fp,"    __m128i zero128;\n");
    fprintf(fp,"    zero128 = _mm_setzero_si128();\n");

    fprintf(fp,"    U0 = (__m256i *)&A256[0];\n");
    fprintf(fp,"    U1 = (__m256i *)&A256[%ld];\n",k);
    fprintf(fp,"    U2 = (__m256i *)&A256[%ld];\n",k*2);
    fprintf(fp,"    V0 = (__m256i *)&B256[0];\n");
    fprintf(fp,"    V1 = (__m256i *)&B256[%ld];\n",k);
    fprintf(fp,"    V2 = (__m256i *)&B256[%ld];\n",k*2);
    
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k);

    // fprintf(fp,"    uint64_t *U1_64 = ((uint64_t *) U1);\n");
    // fprintf(fp,"    uint64_t *V1_64 = ((uint64_t *) V1);\n");
    fprintf(fp,"    __m128i *U1_128 = ((__m128i *) U1);\n");
    fprintf(fp,"    __m128i *V1_128 = ((__m128i *) V1);\n");

    // fprintf(fp,"    W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};\n");
    // fprintf(fp,"    W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};\n");
    fprintf(fp,"    W0[0] = _mm256_set_m128i(U1_128[0],zero128);\n");
    fprintf(fp,"    W4[0] = _mm256_set_m128i(V1_128[0],zero128);\n");
    
    
    // fprintf(fp,"    U1_64 = ((uint64_t *) U1) + 2;\n");
    // fprintf(fp,"    V1_64 = ((uint64_t *) V1) + 2;\n");
    fprintf(fp,"    U1_128 = ((__m128i *) U1) + 1;\n");
    fprintf(fp,"    V1_128 = ((__m128i *) V1) + 1;\n");

    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k-1);
    
    //fprintf(fp,"    int32_t i4 = i << 2;\n");
    fprintf(fp,"        int32_t i2 = i << 1;\n");
    
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    
    //fprintf(fp,"    W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));\n");
    fprintf(fp,"        W0[i1] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);\n");

    fprintf(fp,"        W0[i1] ^= U2[i];\n");
    
    //fprintf(fp,"    W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));\n");
    fprintf(fp,"        W4[i1] = _mm256_set_m128i(V1_128[i2+1],V1_128[i2]);\n");
    
    fprintf(fp,"        W4[i1] ^= V2[i];\n");
    fprintf(fp,"    }\n");
    // fprintf(fp,"    W0[%ld] = (__m256i){U1_64[%ld], U1_64[%ld], 0ul, 0ul} ^ U2[%ld];\n",k,k*4-4,k*4-3,k-1);
    // fprintf(fp,"    W4[%ld] = (__m256i){V1_64[%ld], V1_64[%ld], 0ul, 0ul} ^ V2[%ld];\n",k,k*4-4,k*4-3,k-1);
    fprintf(fp,"    W0[%ld] = _mm256_set_m128i(zero128,U1_128[%ld]) ^ U2[%ld];\n",k,k*2-2,k-1);
    fprintf(fp,"    W4[%ld] = _mm256_set_m128i(zero128,V1_128[%ld]) ^ V2[%ld];\n",k,k*2-2,k-1);

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    W3[%ld] = W0[%ld];\n",k,k);
    fprintf(fp,"    W2[%ld] = W4[%ld];\n",k,k);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld; i++) {\n",k*2 + 2);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+1);
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k);


    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2 + 2);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");

    // fprintf(fp,"    U1_64 = ((uint64_t *) W2) + 2;\n");
    // fprintf(fp,"    uint64_t * U2_64 = ((uint64_t *) W0) + 2;\n");
    fprintf(fp,"    U1_128 = ((__m128i *) W2) + 1;\n");
    fprintf(fp,"    __m128i * U2_128 = ((__m128i *) W0) + 1;\n");


    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",2*k-1);
    
    // fprintf(fp,"    int32_t i4 = i << 2;\n");
    fprintf(fp,"        int32_t i2 = i << 1;\n");
    

    // fprintf(fp,"    W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));\n");
    // fprintf(fp,"    W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));\n");
    fprintf(fp,"        W2[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);\n");
    fprintf(fp,"        W2[i] ^= _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);\n");

    fprintf(fp,"    }\n");

    // fprintf(fp,"    W2[%ld]=(__m256i){U2_64[%ld], U2_64[%ld], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[%ld]));\n",2*k-1,8*k-4,8*k-3,8*k-4);
    // fprintf(fp,"    W2[%ld]=_mm256_lddqu_si256((__m256i const *)(& U1_64[%ld]));\n",2*k,8*k);
    // fprintf(fp,"    W2[%ld]=(__m256i){U1_64[%ld], U1_64[%ld], 0ul, 0ul};\n",2*k+1,8*k+4,8*k+5);
    fprintf(fp,"    W2[%ld]=_mm256_set_m128i(U1_128[%ld],U2_128[%ld]^U1_128[%ld]);\n",2*k-1,4*k-1,4*k-2,4*k-2);
    fprintf(fp,"    W2[%ld]=_mm256_set_m128i(U1_128[%ld],U1_128[%ld]);\n",2*k,4*k+1,4*k);
    fprintf(fp,"    W2[%ld]=_mm256_set_m128i(zero128,U1_128[%ld]);\n",2*k+1,4*k+2);

    // fprintf(fp,"    U1_64 = ((uint64_t *) W4);\n");
    fprintf(fp,"    U1_128 = ((__m128i *) W4);\n");

    fprintf(fp,"    tmp[0] = W2[0] ^ W3[0] ^ W4[0];\n");
    
    // fprintf(fp,"    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};\n");
    fprintf(fp,"    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ _mm256_set_m128i(U1_128[0],zero128);\n");
        
    // fprintf(fp,"    U1_64 = ((uint64_t *) W4) + 2;\n");
    fprintf(fp,"    U1_128 = ((__m128i *) W4) + 1;\n");

    fprintf(fp,"    for(int32_t i = 2; i < %ld; i++) {\n",2*k);
    
    // fprintf(fp,"    int32_t i4 = i << 2;\n");
    // fprintf(fp,"    tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));\n");
    fprintf(fp,"        int32_t i2 = i << 1;\n");
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^_mm256_set_m128i(U1_128[i2 - 3],U1_128[i2 - 4]);\n");
    
    fprintf(fp,"    }\n");

    // fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[%ld]));\n",2*k,2*k,2*k,8*k-8);
    // fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ (__m256i){U1_64[%ld],U1_64[%ld], 0ul, 0ul};\n",2*k+1,2*k+1,2*k+1,8*k-4,8*k-3);
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ _mm256_set_m128i(U1_128[%ld],U1_128[%ld]);\n",2*k,2*k,2*k,4*k-3,4*k-4);
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ _mm256_set_m128i(zero128,U1_128[%ld]);\n",2*k+1,2*k+1,2*k+1,4*k-2);

    fprintf(fp,"    divide_by_x_plus_one_128(W2, tmp, %ld);\n",k*4 + 4);

    // fprintf(fp,"    U1_64 = ((uint64_t *) W3) + 2;\n");
    // fprintf(fp,"    U2_64 = ((uint64_t *) W1) + 2;\n");
    fprintf(fp,"    U1_128 = ((__m128i *) W3) + 1;\n");
    fprintf(fp,"    U2_128 = ((__m128i *) W1) + 1;\n");
    
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k*2 - 1);

    // fprintf(fp,"    int32_t i4 = i << 2;\n");
    // fprintf(fp,"    tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));\n");
    fprintf(fp,"        int32_t i2 = i << 1;\n");
    fprintf(fp,"        tmp[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]) ^ _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);\n");

    fprintf(fp,"    }\n");
    // fprintf(fp,"    tmp[%ld]=(__m256i){U2_64[%ld], U2_64[%ld], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[%ld]));\n",2*k-1,8*k-4,8*k-3,8*k-4);
    // fprintf(fp,"    tmp[%ld]=_mm256_lddqu_si256((__m256i const *)(& U1_64[%ld]));\n",2*k,k*8);
    // fprintf(fp,"    tmp[%ld]=(__m256i){U1_64[%ld], U1_64[%ld], 0ul, 0ul};\n",2*k+1,k*8+4,k*8+5);
    fprintf(fp,"    tmp[%ld]=_mm256_set_m128i(U1_128[%ld], U2_128[%ld]^U1_128[%ld]);\n",2*k-1,4*k-1,4*k-2,4*k-2);
    fprintf(fp,"    tmp[%ld]=_mm256_set_m128i(U1_128[%ld],U1_128[%ld]);\n", 2*k, k*4+1, 4*k);
    fprintf(fp,"    tmp[%ld]=_mm256_set_m128i(zero128, U1_128[%ld]);\n",2*k+1,k*4+2);

    fprintf(fp,"    divide_by_x_plus_one_128(W3, tmp, %ld);\n",4*k+3);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W1[%ld] = W2[%ld];\n",2*k,2*k);
    fprintf(fp,"    W1[%ld] = W2[%ld];\n",2*k+1,2*k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+1);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");
    ///////////////
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++)\n",k);
    fprintf(fp,"    {\n");
    fprintf(fp,"        int32_t j = i + %ld;\n",k);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",k);
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",k*2);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",k*4);
    fprintf(fp,"        Out[j + %ld] = W4[j];\n",k*4);
    fprintf(fp,"    }\n");

    fprintf(fp,"    Out[%ld] ^= W1[%ld];\n",3*k,2*k);
    fprintf(fp,"    Out[%ld] ^= W1[%ld];\n",3*k+1,2*k+1);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",4*k,2*k);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",4*k+1,2*k+1);
    fprintf(fp,"    Out[%ld] ^= W3[%ld];\n",5*k,2*k);
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}
void fprint_TC3_128_3kp1(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp,"static inline void %s(__m256i *Out,   __m256i *A256,   __m256i *B256){\n",func_name);
    uint64_t k=len/3;


    fprintf(fp,"    static __m256i U0[%ld], U1[%ld], U2[%ld], V0[%ld], V1[%ld], V2[%ld];\n",k+1,k+1,k,k+1,k+1,k);
    fprintf(fp,"    static __m256i W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld];\n",k*2+2,k*2+2,k*2+2,k*2+3,k*2);
    fprintf(fp,"    static __m256i tmp[%ld];\n",k*2+3);

    fprintf(fp,"    uint64_t *A = (uint64_t *) A256;\n");
    fprintf(fp,"    uint64_t *B = (uint64_t *) B256;\n");
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        U0[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4]));\n");
    fprintf(fp,"        V0[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4]));\n");
    fprintf(fp,"        U1[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + %ld]));\n",k*4 + 2);
    fprintf(fp,"        V1[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + %ld]));\n",k*4 + 2);
    fprintf(fp,"        U2[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + %ld]));\n",k*8 + 4);
    fprintf(fp,"        V2[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + %ld]));\n",k*8 + 4);
    fprintf(fp,"    }\n");
    fprintf(fp,"    U0[%ld]= (__m256i){A[%ld], A[%ld], 0x0ul, 0x0ul};\n",k,4*k,4*k+1);
    fprintf(fp,"    V0[%ld]= (__m256i){B[%ld], B[%ld], 0x0ul, 0x0ul};\n",k,4*k,4*k+1);
    fprintf(fp,"    U1[%ld]= (__m256i){A[%ld], A[%ld], 0x0ul, 0x0ul};\n",k,k*8 + 2,k*8 + 3);
    fprintf(fp,"    V1[%ld]= (__m256i){B[%ld], B[%ld], 0x0ul, 0x0ul};\n",k,k*8 + 2,k*8 + 3);


    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k,k,k);
    fprintf(fp,"    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k,k,k);

    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k+1);

    fprintf(fp,"    uint64_t *U1_64 = ((uint64_t *) U1);\n");
    fprintf(fp,"    uint64_t *V1_64 = ((uint64_t *) V1);\n");
    fprintf(fp,"    W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};\n");
    fprintf(fp,"    W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) U1) + 2;\n");
    fprintf(fp,"    V1_64 = ((uint64_t *) V1) + 2;\n");

    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));\n");
    fprintf(fp,"        W0[i1] ^= U2[i];\n");
    fprintf(fp,"        W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));\n");
    fprintf(fp,"        W4[i1] ^= V2[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k+1);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k+1);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld; i++) {\n",k*2 + 2);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
//
    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+1);
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k+1);


    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2 + 2);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2+1);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    U1_64 = ((uint64_t *) W2) + 2;\n");
    fprintf(fp,"    uint64_t * U2_64 = ((uint64_t *) W0) + 2;\n");


    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",2*k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));\n");
    fprintf(fp,"        W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld]=(__m256i){U2_64[%ld], U2_64[%ld], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[%ld]));\n",k*2,k*8,k*8+1,k*8);
    fprintf(fp,"    W2[%ld]=(__m256i){U1_64[%ld], U1_64[%ld], 0ul, 0ul};\n",k*2 + 1, k*8+4, k*8+5);


    fprintf(fp,"    U1_64 = ((uint64_t *) W4);\n");
    fprintf(fp,"    tmp[0] = W2[0] ^ W3[0] ^ W4[0];\n");
    fprintf(fp,"    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) W4) + 2;\n");

    fprintf(fp,"    for(int32_t i = 2; i < %ld; i++) {\n",2*k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[%ld]));\n",2*k,2*k,2*k,8*k-8);
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ (__m256i){U1_64[%ld],U1_64[%ld], 0ul, 0ul};\n",2*k+1,2*k+1,2*k+1,8*k-4,8*k-3);
    fprintf(fp,"    divide_by_x_plus_one_128(W2, tmp, %ld);\n",k*4 + 4);

    fprintf(fp,"    U1_64 = ((uint64_t *) W3) + 2;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) W1) + 2;\n");
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k*2);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld]=(__m256i){U2_64[%ld], U2_64[%ld], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[%ld]));\n",k*2, k*8, k*8+1, k*8);
    fprintf(fp,"    tmp[%ld]=(__m256i){U1_64[%ld], U1_64[%ld], 0ul, 0ul};\n",2*k+1,k*8+4,k*8+5);

    fprintf(fp,"    divide_by_x_plus_one_128(W3, tmp, %ld);\n",4*k+3);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W1[%ld] ^= W2[%ld];\n",2*k,2*k);
    fprintf(fp,"    W1[%ld] = W2[%ld];\n",2*k+1,2*k+1);

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+1);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k*2);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[i + %ld] = W2[i];\n",k*2 + 1);
    fprintf(fp,"        Out[i + %ld] = W4[i];\n",k*4 + 2);
    fprintf(fp,"    }\n");

    fprintf(fp,"    Out[%ld] = W0[%ld];\n",k*2,k*2);
    fprintf(fp,"    Out[%ld] = W2[%ld];\n",k*4+1,k*2);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",k*4+2,k*2+1);
    
    fprintf(fp,"    U1_64 = ((uint64_t *) &Out[%ld]) + 2;\n",k);
    fprintf(fp,"    U2_64 = ((uint64_t *) &Out[%ld]) + 2;\n",k*3+1);
    fprintf(fp,"    __m256i aux;\n");
    ////////////
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",2*k+2);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];\n");
    fprintf(fp,"        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);\n");
    fprintf(fp,"        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W3[i];\n");
    fprintf(fp,"        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}
void fprint_TC3_128_3kp2(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp, "static inline void %s(__m256i *Out,   __m256i *A256,   __m256i *B256){\n",func_name);
    uint64_t k=len/3;
    fprintf(fp,"    __m256i *U0, *U1, *U2, *V0, *V1, *V2;\n");

    fprintf(fp,"        static __m256i W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld];\n",k*2+2,k*2+3,k*2+4,k*2+3,k*2);
    fprintf(fp,"        static __m256i tmp[%ld];\n",k*2+4);

    fprintf(fp,"    U0 = (__m256i *)&A256[0];\n");
    fprintf(fp,"    U1 = (__m256i *)&A256[%ld];\n",k+1);
    fprintf(fp,"    U2 = (__m256i *)&A256[%ld];\n",k*2+2);
    fprintf(fp,"    V0 = (__m256i *)&B256[0];\n");
    fprintf(fp,"    V1 = (__m256i *)&B256[%ld];\n",k+1);
    fprintf(fp,"    V2 = (__m256i *)&B256[%ld];\n",k*2+2);

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k,k,k);
    fprintf(fp,"    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k,k,k);
    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k+1);

    fprintf(fp,"    uint64_t *U1_64 = ((uint64_t *) U1);\n");
    fprintf(fp,"    uint64_t *V1_64 = ((uint64_t *) V1);\n");
    fprintf(fp,"    W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};\n");
    fprintf(fp,"    W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) U1) + 2;\n");
    fprintf(fp,"    V1_64 = ((uint64_t *) V1) + 2;\n");

    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));\n");
    fprintf(fp,"        W0[i1] ^= U2[i];\n");
    fprintf(fp,"        W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));\n");
    fprintf(fp,"        W4[i1] ^= V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W0[%ld] = (__m256i){U1_64[%ld], U1_64[%ld], 0ul, 0ul};\n",k+1,k*4,k*4+1);
    fprintf(fp,"    W4[%ld] = (__m256i){V1_64[%ld], V1_64[%ld], 0ul, 0ul};\n",k+1,k*4,k*4+1);

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k+1);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = W0[%ld];\n",k+1,k+1);
    fprintf(fp,"    W2[%ld] = W4[%ld];\n",k+1,k+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k+1);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+2);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld; i++) {\n",k*2 + 3);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
//
    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+2);
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k+1);


    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2 + 3);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2+2);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    U1_64 = ((uint64_t *) W2) + 2;\n");
    fprintf(fp,"    uint64_t * U2_64 = ((uint64_t *) W0) + 2;\n");


    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",2*k+1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));\n");
    fprintf(fp,"        W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld]=(__m256i){U1_64[%ld]^U2_64[%ld], U1_64[%ld]^U2_64[%ld], U1_64[%ld], U1_64[%ld]};\n",2*k+1,8*k+4,8*k+4,8*k+5,8*k+5,8*k+6,8*k+7);
    fprintf(fp,"    W2[%ld]=(__m256i){U1_64[%ld], U1_64[%ld], 0ul, 0ul};\n",2*k+2,8*k+8,8*k+9);

    fprintf(fp,"    U1_64 = ((uint64_t *) W4);\n");
    fprintf(fp,"    tmp[0] = W2[0] ^ W3[0] ^ W4[0];\n");
    fprintf(fp,"    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) W4) + 2;\n");

    fprintf(fp,"    for(int32_t i = 2; i < %ld; i++) {\n",2*k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[%ld]));\n",2*k,2*k,2*k,8*k-8);
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ (__m256i){U1_64[%ld],U1_64[%ld], 0ul, 0ul};\n",2*k+1,2*k+1,2*k+1,8*k-4,8*k-3);
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld];\n",2*k+2,2*k+2,2*k+2);
    fprintf(fp,"    divide_by_x_plus_one_128(W2, tmp, %ld);\n",k*4 + 6);
//
    fprintf(fp,"    U1_64 = ((uint64_t *) W3) + 2;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) W1) + 2;\n");
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k*2 + 1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld]=(__m256i){U1_64[%ld]^U2_64[%ld], U1_64[%ld]^U2_64[%ld], U1_64[%ld], U1_64[%ld]};\n",2*k+1,k*8+4,k*8+4,k*8+5,k*8+5,k*8+6,k*8+7);
    fprintf(fp,"    tmp[%ld]=(__m256i){U1_64[%ld], U1_64[%ld], 0ul, 0ul};\n",2*k+2,k*8+8,k*8+9);
    fprintf(fp,"    divide_by_x_plus_one_128(W3, tmp, %ld);\n",4*k+5);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W1[%ld] ^= W2[%ld];\n",2*k,2*k);
    fprintf(fp,"    W1[%ld] ^= W2[%ld];\n",2*k+1,2*k+1);
    fprintf(fp,"    W1[%ld] = W2[%ld];\n",2*k+2,2*k+2);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");
    ///////////////
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++)\n",k-1);
    fprintf(fp,"    {\n");
    fprintf(fp,"        int32_t j = i + %ld;\n",k+1);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",k+1);
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",k*2+2);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",k*4+4);
    fprintf(fp,"        Out[j + %ld] = W4[j];\n",k*4+4);
    fprintf(fp,"    }\n");

    fprintf(fp,"    for (int32_t i = %ld; i < %ld; i++)\n",k-1,k+1);
    fprintf(fp,"    {\n");
    fprintf(fp,"        int32_t j = i + %ld;\n",k+1);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",k+1);
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",k*2+2);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",k*4+4);
    fprintf(fp,"    }\n");


    fprintf(fp,"    Out[%ld] ^= W1[%ld];\n",3*k+3,2*k+2);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",4*k+4,2*k+2);
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}

void fprint_TC3_64_3k(FILE * fp, char * func_name, uint64_t len){
    //사실 64 
    fprintf(fp, "static inline void %s(__m256i *Out,   __m256i *A256,   __m256i *B256){\n",func_name);
    
    uint64_t k=len/3;
    fprintf(fp,"    __m256i *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp,"    static __m256i W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld];\n",k*2,k*2+1,k*2+2,k*2+1,k*2);
    fprintf(fp,"    static  __m256i tmp[%ld];\n",k*2+2);
    fprintf(fp,"    U0 = (__m256i *)&A256[0];\n");
    fprintf(fp,"    U1 = (__m256i *)&A256[%ld];\n",k);
    fprintf(fp,"    U2 = (__m256i *)&A256[%ld];\n",2*k);
    fprintf(fp,"    V0 = (__m256i *)&B256[0];\n");
    fprintf(fp,"    V1 = (__m256i *)&B256[%ld];\n",k);
    fprintf(fp,"    V2 = (__m256i *)&B256[%ld];\n",2*k);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k);
    fprintf(fp,"    uint64_t *U1_64 = ((uint64_t *) U1);\n");
    fprintf(fp,"    uint64_t *U2_64 = ((uint64_t *) U2);\n");
    fprintf(fp,"    uint64_t *V1_64 = ((uint64_t *) V1);\n");
    fprintf(fp,"    uint64_t *V2_64 = ((uint64_t *) V2);\n");
    fprintf(fp,"    W0[0] = (__m256i){0ul, U1_64[0], U1_64[1] ^ U2_64[0], U1_64[2] ^ U2_64[1]};\n");
    fprintf(fp,"    W4[0] = (__m256i){0ul, V1_64[0], V1_64[1] ^ V2_64[0], V1_64[2] ^ V2_64[1]};\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) U1) + 3;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) U2) + 2;\n");
    fprintf(fp,"    V1_64 = ((uint64_t *) V1) + 3;\n");
    fprintf(fp,"    V2_64 = ((uint64_t *) V2) + 2;\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k-1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W0[i1] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));\n");
    fprintf(fp,"        W0[i1] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));\n");
    fprintf(fp,"        W4[i1] = _mm256_lddqu_si256((__m256i   *)(& V1_64[i4]));\n");
    fprintf(fp,"        W4[i1] ^= _mm256_lddqu_si256((__m256i   *)(& V2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W0[%ld] = (__m256i){U1_64[%ld]^U2_64[%ld], U2_64[%ld], 0ul, 0ul};\n",k,4*k-4,4*k-4,4*k-3);
    fprintf(fp,"    W4[%ld] = (__m256i){V1_64[%ld]^V2_64[%ld], V2_64[%ld], 0ul, 0ul};\n",k,4*k-4,4*k-4,4*k-3);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"        W3[%ld] = W0[%ld];\n",k,k);
    fprintf(fp,"        W2[%ld] = W4[%ld];\n",k,k);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+1);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+1);
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) W2) + 1;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) W0) + 1;\n");
    fprintf(fp,"    U1_64[%ld]=0;\n",8*k+3);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k-1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        W2[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));\n");
    fprintf(fp,"        W2[i] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld]=(__m256i){U1_64[%ld]^U2_64[%ld], U1_64[%ld]^U2_64[%ld], U1_64[%ld]^U2_64[%ld], U1_64[%ld]};\n",k*2 - 1,k*8-4,k*8-4,k*8-3,k*8-3,k*8-2,k*8-2,k*8-1);
    fprintf(fp,"    W2[%ld]=_mm256_lddqu_si256((__m256i   *)(& U1_64[%ld]));\n",2*k,8*k);
    fprintf(fp,"    U1_64 = ((uint64_t *) W4);\n");
    fprintf(fp,"    tmp[0] = W2[0] ^ W3[0] ^ W4[0] ^ (__m256i){0x0ul, 0x0ul, 0x0ul, U1_64[0]};\n");
    fprintf(fp,"    U1_64++;\n");
    fprintf(fp,"    for (int32_t i = 1; i < %ld; i++){\n",2*k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i   *)(& U1_64[i4 - 4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ (__m256i){U1_64[%ld],U1_64[%ld],U1_64[%ld], 0x0ul};\n",2*k,2*k,2*k,8*k-4,8*k-3,8*k-2);
    fprintf(fp,"    U1_64 = ((uint64_t *) W2);\n");
    fprintf(fp,"    U1_64[%ld]=0;\n",8*k+3);
    fprintf(fp,"    divide_by_x_plus_one_64(tmp, tmp, %ld);\n",8*k+3);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        W2[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) W3) + 1;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) W1) + 1;\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k-1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld]=(__m256i){U1_64[%ld]^U2_64[%ld],U1_64[%ld]^U2_64[%ld],U1_64[%ld]^U2_64[%ld],U1_64[%ld]};\n",2*k-1,k*8-4,k*8-4,k*8-3,k*8-3,k*8-2,k*8-2,k*8-1);
    fprintf(fp,"    tmp[%ld]=(__m256i){U1_64[%ld],U1_64[%ld],U1_64[%ld],0x0ul};\n",2*k,8*k,8*k+1,8*k+2);
    fprintf(fp,"    U1_64[%ld]=0;\n",8*k+2);
    fprintf(fp,"    divide_by_x_plus_one_64(tmp, tmp, %ld);\n",8*k+3);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W1[%ld]=W2[%ld];\n",2*k,2*k);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");        
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k);
    fprintf(fp,"        int32_t j = i + %ld;\n",k);
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",k);
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k);
    fprintf(fp,"        Out[j + %ld] = W4[j];\n",4*k);
    fprintf(fp,"    }\n");
    fprintf(fp,"        Out[%ld] ^= W1[%ld];\n",3*k,2*k);
    fprintf(fp,"        Out[%ld] ^= W2[%ld];\n",4*k,2*k);
    fprintf(fp,"        Out[%ld] ^= W3[%ld];\n",5*k,2*k);
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}
void fprint_TC3_64_3kp1(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp, "static inline void %s(__m256i *Out,   __m256i *A256,   __m256i *B256){\n",func_name);
    uint64_t k=len/3;
    fprintf(fp,"    static __m256i UV[%ld];\n",6*k+6);
    fprintf(fp,"    static __m256i *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp,"    static __m256i W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld];\n",k*2+2,k*2+2,k*2+2,k*2+2,k*2+2);
    fprintf(fp,"    static  __m256i tmp[%ld];\n",k*6+4);
    fprintf(fp,"    static __m256i zero = {0ul, 0ul, 0ul, 0ul};\n");

    fprintf(fp,"    uint64_t *A = (uint64_t *) A256;\n");
    fprintf(fp,"    uint64_t *B = (uint64_t *) B256;\n");

    fprintf(fp,"    U0 = UV;\n");
    fprintf(fp,"    U1 = U0 + %ld;\n",k+1);
    fprintf(fp,"    U2 = U1 + %ld;\n",k+1);
    fprintf(fp,"    V0 = U2 + %ld;\n",k+1);
    fprintf(fp,"    V1 = V0 + %ld;\n",k+1);
    fprintf(fp,"    V2 = V1 + %ld;\n",k+1);
    
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        U0[i]= _mm256_lddqu_si256((__m256i   *)(& A[i4]));\n");
    fprintf(fp,"        V0[i]= _mm256_lddqu_si256((__m256i   *)(& B[i4]));\n");
    fprintf(fp,"        U1[i]= _mm256_lddqu_si256((__m256i   *)(& A[i4 + %ld]));\n",k*4 + 2);
    fprintf(fp,"        V1[i]= _mm256_lddqu_si256((__m256i   *)(& B[i4 + %ld]));\n",k*4 + 2);
    fprintf(fp,"        U2[i]= _mm256_lddqu_si256((__m256i   *)(& A[i4 + %ld]));\n",k*8 + 4);
    fprintf(fp,"        V2[i]= _mm256_lddqu_si256((__m256i   *)(& B[i4 + %ld]));\n",k*8 + 4);
    fprintf(fp,"    }\n");
    fprintf(fp,"    U0[%ld]= (__m256i){A[%ld], A[%ld], 0x0ul, 0x0ul};\n",k,k*4,k*4+1);
    fprintf(fp,"    V0[%ld]= (__m256i){B[%ld], B[%ld], 0x0ul, 0x0ul};\n",k,k*4,k*4+1);
    fprintf(fp,"    U1[%ld]= (__m256i){A[%ld], A[%ld], 0x0ul, 0x0ul};\n",k,k*8+2,k*8+3);
    fprintf(fp,"    V1[%ld]= (__m256i){B[%ld], B[%ld], 0x0ul, 0x0ul};\n",k,k*8+2,k*8+3);
    fprintf(fp,"    U2[%ld]= zero;\n",k);
    fprintf(fp,"    V2[%ld]= zero;\n",k);

    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k,k,k);
    fprintf(fp,"    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k,k,k);
    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k+1);
    fprintf(fp,"    uint64_t *U1_64 = ((uint64_t *) U1);\n");
    fprintf(fp,"    uint64_t *U2_64 = ((uint64_t *) U2);\n");
    fprintf(fp,"    uint64_t *V1_64 = ((uint64_t *) V1);\n");
    fprintf(fp,"    uint64_t *V2_64 = ((uint64_t *) V2);\n");
    fprintf(fp,"    W0[0] = (__m256i){0ul, U1_64[0], U1_64[1] ^ U2_64[0], U1_64[2] ^ U2_64[1]};\n");
    fprintf(fp,"    W4[0] = (__m256i){0ul, V1_64[0], V1_64[1] ^ V2_64[0], V1_64[2] ^ V2_64[1]};\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) U1) + 3;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) U2) + 2;\n");
    fprintf(fp,"    V1_64 = ((uint64_t *) V1) + 3;\n");
    fprintf(fp,"    V2_64 = ((uint64_t *) V2) + 2;\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W0[i1] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));\n");
    fprintf(fp,"        W0[i1] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));\n");
    fprintf(fp,"        W4[i1] = _mm256_lddqu_si256((__m256i   *)(& V1_64[i4]));\n");
    fprintf(fp,"        W4[i1] ^= _mm256_lddqu_si256((__m256i   *)(& V2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k+1);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k+1);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+1);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+1);
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k+1);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) W2) + 1;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) W0) + 1;\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        W2[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));\n");
    fprintf(fp,"        W2[i] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld]=(__m256i){U1_64[%ld], 0x0ul, 0x0ul, 0x0ul};\n",k*2 + 1,k*8+4);
    fprintf(fp,"    U1_64 = ((uint64_t *) W4);\n");
    fprintf(fp,"    tmp[0] = W2[0] ^ W3[0] ^ W4[0] ^ (__m256i){0x0ul, 0x0ul, 0x0ul, U1_64[0]};\n");
    fprintf(fp,"    U1_64++;\n");
    fprintf(fp,"    for (int32_t i = 1; i < %ld; i++){\n",2*k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i   *)(& U1_64[i4 - 4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ (__m256i){U1_64[%ld],U1_64[%ld],U1_64[%ld], 0x0ul};\n",2*k,2*k,2*k,8*k-4,8*k-3,8*k-2);
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld];\n",2*k+1,2*k+1,2*k+1);
    fprintf(fp,"    W2[%ld]=zero;\n",2*k+1);
    fprintf(fp,"    divide_by_x_plus_one_64(tmp, tmp, %ld);\n",8*k+6);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W2[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) W3) + 1;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) W1) + 1;\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld]=(__m256i){U1_64[%ld],0x0ul,0x0ul,0x0ul};\n",2*k+1,8*k+4);
    fprintf(fp,"    W3[%ld]=zero;\n",2*k+1);
    fprintf(fp,"    divide_by_x_plus_one_64(tmp, tmp, %ld);\n",8*k+5);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W1[%ld] ^= W2[%ld];\n",2*k,2*k);
    fprintf(fp,"    W1[%ld] = W2[%ld];\n",2*k+1,2*k+1);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k);
    fprintf(fp,"        tmp[i] = W0[i];\n");
    fprintf(fp,"        tmp[i + %ld] = W2[i];\n",2*k+1);
    fprintf(fp,"        tmp[i + %ld] = W4[i];\n",4*k+2);
    fprintf(fp,"    }\n");
    fprintf(fp,"        tmp[%ld] = W0[%ld];\n",2*k,2*k);
    fprintf(fp,"        tmp[%ld] = W2[%ld];\n",4*k+1,2*k);
    fprintf(fp,"        tmp[%ld] ^= W2[%ld];\n",4*k+2,2*k+1);
    fprintf(fp,"    U1_64 = ((uint64_t *) &tmp[%ld]) + 2;\n",k);
    fprintf(fp,"    U2_64 = ((uint64_t *) &tmp[%ld]) + 2;\n",k*3+1);
    fprintf(fp,"    __m256i aux;\n");
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",2*k+1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];\n");
    fprintf(fp,"        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);\n");
    fprintf(fp,"        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W3[i];\n");
    fprintf(fp,"        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[%ld])) ^ W1[%ld];\n",8*k+4,2*k+1);
    fprintf(fp,"    _mm256_storeu_si256 ((__m256i *) (& U1_64[%ld]), aux);\n",8*k+4);
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",6*k+2);
    fprintf(fp,"        Out[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}
void fprint_TC3_64_3kp2(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp, "static inline void %s(__m256i *Out,   __m256i *A256,   __m256i *B256){\n",func_name);
    uint64_t k=len/3;
    fprintf(fp,"    static __m256i UV[%ld];\n",6*k+6);
    fprintf(fp,"    static __m256i *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp,"    static __m256i W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld];\n",k*2+2,k*2+2,k*2+2,k*2+2,k*2+2);
    fprintf(fp,"    static  __m256i tmp[%ld];\n",k*6+4);

    fprintf(fp,"    uint64_t *A = (uint64_t *) A256;\n");
    fprintf(fp,"    uint64_t *B = (uint64_t *) B256;\n");
    
    fprintf(fp,"    U0 = UV;\n");
    fprintf(fp,"    U1 = U0 + %ld;\n",k+1);
    fprintf(fp,"    U2 = U1 + %ld;\n",k+1);
    fprintf(fp,"    V0 = U2 + %ld;\n",k+1);
    fprintf(fp,"    V1 = V0 + %ld;\n",k+1);
    fprintf(fp,"    V2 = V1 + %ld;\n",k+1);

    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        U0[i]= _mm256_lddqu_si256((__m256i   *)(& A[i4]));\n");
    fprintf(fp,"        V0[i]= _mm256_lddqu_si256((__m256i   *)(& B[i4]));\n");
    fprintf(fp,"        U1[i]= _mm256_lddqu_si256((__m256i   *)(& A[i4 + %ld]));\n",k*4 + 3);
    fprintf(fp,"        V1[i]= _mm256_lddqu_si256((__m256i   *)(& B[i4 + %ld]));\n",k*4 + 3);
    fprintf(fp,"        U2[i]= _mm256_lddqu_si256((__m256i   *)(& A[i4 + %ld]));\n",k*8 + 6);
    fprintf(fp,"        V2[i]= _mm256_lddqu_si256((__m256i   *)(& B[i4 + %ld]));\n",k*8 + 6);
    fprintf(fp,"    }\n");
    fprintf(fp,"    U0[%ld]= (__m256i){A[%ld], A[%ld], A[%ld], 0x0ul};\n",k,k*4,k*4+1,k*4+2);
    fprintf(fp,"    V0[%ld]= (__m256i){B[%ld], B[%ld], B[%ld], 0x0ul};\n",k,k*4,k*4+1,k*4+2);
    fprintf(fp,"    U1[%ld]= (__m256i){A[%ld], A[%ld], A[%ld], 0x0ul};\n",k,k*8+3,k*8+4,k*8+5);
    fprintf(fp,"    V1[%ld]= (__m256i){B[%ld], B[%ld], B[%ld], 0x0ul};\n",k,k*8+3,k*8+4,k*8+5);
    fprintf(fp,"    U2[%ld]= (__m256i){A[%ld], A[%ld], 0x0ul, 0x0ul};\n",k,12*k+6,12*k+7);
    fprintf(fp,"    V2[%ld]= (__m256i){B[%ld], B[%ld], 0x0ul, 0x0ul};\n",k,12*k+6,12*k+7);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k+1);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k+1);
    fprintf(fp,"    uint64_t *U1_64 = ((uint64_t *) U1);\n");
    fprintf(fp,"    uint64_t *U2_64 = ((uint64_t *) U2);\n");
    fprintf(fp,"    uint64_t *V1_64 = ((uint64_t *) V1);\n");
    fprintf(fp,"    uint64_t *V2_64 = ((uint64_t *) V2);\n");
    fprintf(fp,"    W0[0] = (__m256i){0ul, U1_64[0], U1_64[1] ^ U2_64[0], U1_64[2] ^ U2_64[1]};\n");
    fprintf(fp,"    W4[0] = (__m256i){0ul, V1_64[0], V1_64[1] ^ V2_64[0], V1_64[2] ^ V2_64[1]};\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) U1) + 3;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) U2) + 2;\n");
    fprintf(fp,"    V1_64 = ((uint64_t *) V1) + 3;\n");
    fprintf(fp,"    V2_64 = ((uint64_t *) V2) + 2;\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W0[i1] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));\n");
    fprintf(fp,"        W0[i1] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));\n");
    fprintf(fp,"        W4[i1] = _mm256_lddqu_si256((__m256i   *)(& V1_64[i4]));\n");
    fprintf(fp,"        W4[i1] ^= _mm256_lddqu_si256((__m256i   *)(& V2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k+1);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",k+1);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+1);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+1);
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k+1);
    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k+1);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) W2) + 1;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) W0) + 1;\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        W2[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));\n");
    fprintf(fp,"        W2[i] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld]=(__m256i){U1_64[%ld]^U2_64[%ld], U1_64[%ld]^U2_64[%ld], U1_64[%ld]^U2_64[%ld], 0x0ul};\n",k*2 + 1,k*8+4,k*8+4,k*8+5,k*8+5,k*8+6,k*8+6);
    fprintf(fp,"    U1_64 = ((uint64_t *) W4);\n");
    fprintf(fp,"    tmp[0] = W2[0] ^ W3[0] ^ W4[0] ^ (__m256i){0x0ul, 0x0ul, 0x0ul, U1_64[0]};\n");
    fprintf(fp,"    U1_64++;\n");
    fprintf(fp,"    for (int32_t i = 1; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i   *)(& U1_64[i4 - 4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ (__m256i){U1_64[%ld],U1_64[%ld],U1_64[%ld], 0x0ul};\n",2*k+1,2*k+1,2*k+1,8*k,8*k+1,8*k+2);
    fprintf(fp,"    divide_by_x_plus_one_64(tmp, tmp, %ld);\n",8*k+8);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W2[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    U1_64 = ((uint64_t *) W3) + 1;\n");
    fprintf(fp,"    U2_64 = ((uint64_t *) W1) + 1;\n");
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        tmp[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld]=(__m256i){U1_64[%ld]^U2_64[%ld],U1_64[%ld]^U2_64[%ld],U1_64[%ld]^U2_64[%ld],0x0ul};\n",2*k+1,8*k+4,8*k+4,8*k+5,8*k+5,8*k+6,8*k+6);
    fprintf(fp,"    divide_by_x_plus_one_64(tmp, tmp, %ld);\n",8*k+7);
    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+2);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");

    fprintf(fp,"        memset((__m256i*)tmp, 0, sizeof(__m256i) * %ld);\n",6*k+4);

    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++){\n",2*k+1);
    fprintf(fp,"        tmp[i] = W0[i];\n");
    fprintf(fp,"        tmp[i + %ld] = W4[i];\n",4*k+3);
    fprintf(fp,"    }\n");
    fprintf(fp,"        tmp[%ld] = W0[%ld];\n",2*k+1,2*k+1);

    fprintf(fp,"    U1_64 = ((uint64_t *) &tmp[%ld]) + 3;\n",k);
    fprintf(fp,"    U2_64 = ((uint64_t *) &tmp[%ld]) + 2;\n",k*2+1);
    fprintf(fp,"    V1_64 = ((uint64_t *) &tmp[%ld]) + 1;\n",k*3+2);
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",2*k+2);
    fprintf(fp,"        int32_t i4 = i << 2;\n");
    fprintf(fp,"        __m256i aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];\n");
    fprintf(fp,"        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);\n");
    fprintf(fp,"        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W2[i];\n");
    fprintf(fp,"        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);\n");
    fprintf(fp,"        aux = _mm256_lddqu_si256 ((__m256i *) (& V1_64[i4])) ^ W3[i];\n");
    fprintf(fp,"        _mm256_storeu_si256 ((__m256i *) (& V1_64[i4]), aux);\n");
    fprintf(fp,"    }\n");
    
    fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",6*k+4);
    fprintf(fp,"        Out[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
    

}
//#endif

#if defined(BEST_ALG_AVX2_PCLMUL)
uint8_t best_algorithm[MAX_LEN+1] = BEST_ALG_AVX2_PCLMUL;
#elif defined(BEST_ALG_AVX2_VPCLMUL)
uint8_t best_algorithm[MAX_LEN+1] = BEST_ALG_AVX2_VPCLMUL;
#elif defined(BEST_ALG_NEON)
uint8_t best_algorithm[MAX_LEN+1] = BEST_ALG_NEON;
#elif defined(BEST_ALG_new)
uint8_t best_algorithm[MAX_LEN+1] = BEST_ALG_new;
#else
uint8_t best_algorithm[MAX_LEN+1] = {0};
#endif

void best_alg_set(uint32_t dimension, uint8_t alg_num, uint8_t env_num){
    FILE * fp1;
    switch (env_num)
    {
    case 1: //AVX2 + PCLMULQDQ
        fp1 = fopen("best_alg/best_alg_result_AVX2_PCLMUL.h","w");
        fprintf(fp1, "#define BEST_ALG_AVX2_PCLMUL { \\\n");
        break;
    case 2: //AVX2 + VPCLMULQDQ
        fp1 = fopen("best_alg/best_alg_result_AVX2_VPCLMUL.h","w");
        fprintf(fp1, "#define BEST_ALG_AVX2_VPCLMUL { \\\n");
        break;
    case 3: //NEON
        fp1 = fopen("best_alg/best_alg_result_NEON.h","w");
        fprintf(fp1, "#define BEST_ALG_NEON { \\\n");
        break;    
    default:
        fp1 = fopen("best_alg/best_alg_result_new.h","w");
        fprintf(fp1, "#define BEST_ALG_new { \\\n");
        break;
    }
    best_algorithm[dimension] = alg_num;
    for(int i=0;i<MAX_LEN+1;i++){
        fprintf(fp1, "%d, ", best_algorithm[i]);
        if(i%20 == 19 || i == MAX_LEN) fprintf(fp1, " \\\n");
    }
    fprintf(fp1, "}\n");
    fprintf(fp1, "\n");
    fclose(fp1);
}
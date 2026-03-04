/**
 * @file gf2x.c
 * @brief Implementation of multiplication of two polynomials
 */

#include "gf2x.h"
#include "parameters.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <arm_neon.h>

#define VEC_N_ARRAY_SIZE_VEC CEIL_DIVIDE(PARAM_N_MULT, 256) /*!< The number of needed vectors to store PARAM_N bits*/
#define WORD 64
#define LAST64 (PARAM_N >> 6)
uint64_t o256[VEC_N_ARRAY_SIZE_VEC << 2];

 /**
  * @brief Compute o(x) = a(x) mod \f$ X^n - 1\f$
  *
  * This function computes the modular reduction of the polynomial a(x)
  *
  * @param[out] o Pointer to the result
  * @param[in] a Pointer to the polynomial a(x)
  */
void reduce(uint64_t *o, const uint64_t *a) {
    uint64_t r, carry;
    static const int32_t dec64 = PARAM_N & 0x3f;
    static int32_t d0;
    int32_t i, i2;

    d0 = WORD - dec64;
    for (i = LAST64 ; i < (PARAM_N >> 5) - 4; i += 1) {
       r =  a[i] >> dec64;
       carry = a[i+1] << d0;  
   
       r ^= carry;
       i2 = (i - LAST64);
       o256[i2] = a[i2] ^ r;
   }
   i = i - LAST64;

   for (; i < LAST64 + 1 ; i++) {
       r = a[i + LAST64] >> dec64;
       carry = a[i + LAST64 + 1] << d0;
       r ^= carry;
       o256[i] = a[i] ^ r;
   }

   o256[LAST64] &= BITMASK(PARAM_N, 64);
   memcpy(o, o256, VEC_N_SIZE_BYTES);
}

// static inline void schmul64_NEON(poly8x16_t* res, poly8x8_t* a, poly8x8_t* b){
//     poly8x8_t a_shift, b_shift;
//     poly8x16_t poly_tmp1, poly_tmp2, poly_res;
//     poly8x8_t *p1, *p2;
//     uint8x8_t mask1={0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0,0};
//     uint8x8_t mask2={0xFF,0xFF,0xFF,0xFF,    0,   0,0,0};
//     uint8x8_t mask3={0xFF,0xFF,   0,   0,    0,   0,0,0};
//     
//     poly_res=vreinterpretq_p8_p16(vmull_p8(*a,*b));
//     
//     a_shift=vext_p8(*a,*a,1);
//     b_shift=vext_p8(*b,*b,1);
//     poly_tmp1=vreinterpretq_p8_p16(vmull_p8(*a,b_shift));
//     poly_tmp2=vreinterpretq_p8_p16(vmull_p8(a_shift,*b));
//     poly_tmp1=vaddq_p8(poly_tmp1,poly_tmp2);
//     p1=(poly8x8_t*)&poly_tmp1;
//     p2=((poly8x8_t*)&poly_tmp1)+1;
//     *p1=vadd_p8(*p1, *p2);
//     
//     *p2=vreinterpret_p8_u8(vand_u8(vreinterpret_u8_p8(*p2), mask1));
//     *p1=vadd_p8(*p1, *p2);
//     poly_tmp1=vextq_p8(poly_tmp1,poly_tmp1,15);
// 
//     poly_res=vaddq_p8(poly_res, poly_tmp1);
// 
//     a_shift=vext_p8(a_shift,a_shift,1);
//     b_shift=vext_p8(b_shift,b_shift,1);
//     poly_tmp1=vreinterpretq_p8_p16(vmull_p8(*a,b_shift));
//     poly_tmp2=vreinterpretq_p8_p16(vmull_p8(a_shift,*b));
//     poly_tmp1=vaddq_p8(poly_tmp1,poly_tmp2);
//     *p1=vadd_p8(*p1, *p2);
//     *p2=vreinterpret_p8_u8(vand_u8(vreinterpret_u8_p8(*p2), mask2));
//     *p1=vadd_p8(*p1, *p2);
//     poly_tmp1=vextq_p8(poly_tmp1,poly_tmp1,14);
//     
//     poly_res=vaddq_p8(poly_res, poly_tmp1);
//     
//     a_shift=vext_p8(a_shift,a_shift,1);
//     b_shift=vext_p8(b_shift,b_shift,1);
//     
//     poly_tmp1=vreinterpretq_p8_p16(vmull_p8(*a,b_shift));
//     poly_tmp2=vreinterpretq_p8_p16(vmull_p8(a_shift,*b));
//     poly_tmp1=vaddq_p8(poly_tmp1,poly_tmp2);
//     *p1=vadd_p8(*p1, *p2);
//     *p2=vreinterpret_p8_u8(vand_u8(vreinterpret_u8_p8(*p2), mask3));
//     *p1=vadd_p8(*p1, *p2);
//     poly_tmp1=vextq_p8(poly_tmp1,poly_tmp1,13);
//     
//     poly_res=vaddq_p8(poly_res, poly_tmp1);
// 
//     b_shift=vext_p8(b_shift,b_shift,1);
//     poly_tmp1=vreinterpretq_p8_p16(vmull_p8(*a,b_shift));
//     *p1=vadd_p8(*p1, *p2);
//     *p2=vadd_p8(*p2, *p2);
//     poly_tmp1=vextq_p8(poly_tmp1,poly_tmp1,12);
//     
//     *res = vaddq_p8(poly_res, poly_tmp1);
// }

static inline void divide_by_x_plus_one_128(poly8x16_t *in, poly8x16_t *out, int32_t size) {
    out[0] = in[0];
    for(int32_t i = 1 ; i < size ; i++) {
        out[i] = out[i - 1] ^ in[i];
    }
}

static inline void divide_by_x_plus_one_64(poly8x16_t *out,poly8x16_t *in,int32_t size){
    uint64_t *A = (uint64_t *) in;
    uint64_t *B = (uint64_t *) out;

    B[0] = A[0];
    for(int32_t i = 1; i < size; i++) {
        B[i] = B[i - 1] ^ A[i];
    }
}

static inline poly128_t vmull64_a72(const uint64_t a_64, const uint64_t b_64)
{
    const poly8x8_t a = vreinterpret_p8_u64(vdup_n_u64(a_64));

    const poly8x8_t b = vreinterpret_p8_u64(vdup_n_u64(b_64));

    poly8x8_t a_1 = vext_p8(a, a, 1);
    poly8x8_t a_2 = vext_p8(a, a, 2);
    poly8x8_t a_3 = vext_p8(a, a, 3);
    poly8x8_t b_1 = vext_p8(b, b, 1);
    poly8x8_t b_2 = vext_p8(b, b, 2);
    poly8x8_t b_3 = vext_p8(b, b, 3);
    poly8x8_t b_4 = vext_p8(b, b, 4);
    poly16x8_t d = vmull_p8(a, b);
    uint64_t k_48 = 0x0000FFFFFFFFFFFF;
    poly16x8_t L = vaddq_p16(vmull_p8(a, b_1), vmull_p8(a_1, b));
    poly16x4_t result_high = vget_high_p16(L);
    poly16x4_t result_low = vget_low_p16(L);
    result_low = vadd_p16(result_high, result_low);
    result_high = vreinterpret_p16_u16(vand_u16(vreinterpret_u16_p16(result_high), vreinterpret_u16_u64(vdup_n_u64(k_48))));
    result_low = vadd_p16(result_high, result_low);
    poly8x16_t temp_result = vreinterpretq_p8_p16(vcombine_p16(result_low, result_high));
    temp_result = vextq_p8(temp_result, temp_result, 15);


    poly16x8_t result = vaddq_p16( d, vreinterpretq_p16_p8(temp_result));


    poly16x8_t M = vaddq_p16(vmull_p8(a, b_2), vmull_p8(a_2, b));
    uint64_t k_32 = 0x00000000FFFFFFFF;
    result_high = vget_high_p16(M);
    result_low = vget_low_p16(M);
    result_low = vadd_p16(result_high, result_low);
    result_high = vreinterpret_p16_u16(vand_u16(vreinterpret_u16_p16(result_high), vreinterpret_u16_u64(vdup_n_u64(k_32))));
    result_low = vadd_p16(result_high, result_low);
    temp_result = vreinterpretq_p8_p16(vcombine_p16(result_low, result_high));
    temp_result = vextq_p8(temp_result, temp_result, 14);
    result = vaddq_p16(result, vreinterpretq_p16_p8(temp_result));

    poly16x8_t N = vaddq_p16(vmull_p8(a, b_3), vmull_p8(a_3, b));
    uint64_t k_16 = 0x000000000000FFFF;
    result_high = vget_high_p16(N);
    result_low = vget_low_p16(N);
    result_low = vadd_p16(result_high, result_low);
    result_high = vreinterpret_p16_u16(vand_u16(vreinterpret_u16_p16(result_high), vreinterpret_u16_u64(vdup_n_u64(k_16))));
    result_low = vadd_p16(result_high, result_low);
    temp_result = vreinterpretq_p8_p16(vcombine_p16(result_low, result_high));
    temp_result = vextq_p8(temp_result, temp_result, 13);
    result = vaddq_p16(result, vreinterpretq_p16_p8(temp_result));

    poly16x8_t K = vmull_p8(a, b_4);
    result_high = vget_high_p16(K);
    result_low = vget_low_p16(K);
    result_low = vadd_p16(result_high, result_low);
    result_high = vdup_n_p16(0);
    temp_result = vreinterpretq_p8_p16(vcombine_p16(result_low, result_high));
    temp_result = vextq_p8(temp_result, temp_result, 12);
    result = vaddq_p16(result, vreinterpretq_p16_p8(temp_result));
    return vreinterpretq_p128_p16(result);
}

//len = 0: karat_NEON
static inline void gfmul_1(poly8x16_t* Out, poly8x16_t* a, poly8x16_t* b){
	poly8x8_t al, ah, bl, bh;
	poly8x8_t alpah, blpbh;
	poly8x16_t D0={0}, D1={0}, D2={0};

	ah = vget_high_p8(*a);
	al = vget_low_p8(*a);
	bh = vget_high_p8(*b);
	bl = vget_low_p8(*b);

	alpah=vadd_p8(al,ah);
	blpbh=vadd_p8(bl,bh);
    // schmul64_NEON(&D1, &alpah, &blpbh);
    // schmul64_NEON(&D0, &al, &bl);
    // schmul64_NEON(&D2, &ah, &bh);
    D1 = (poly8x16_t)vmull64_a72(*(uint64_t *)&alpah, *(uint64_t *)&blpbh);
    D0 = (poly8x16_t)vmull64_a72(*(uint64_t *)&al, *(uint64_t *)&bl);
    D2 = (poly8x16_t)vmull64_a72(*(uint64_t *)&ah, *(uint64_t *)&bh);

    D1=vaddq_p8(D1,D0);
    D1=vaddq_p8(D1,D2);

    Out[0]=vcombine_p8(vget_low_p8(D0),vadd_p8(vget_high_p8(D0),vget_low_p8(D1)));
    Out[1]=vcombine_p8(vadd_p8(vget_low_p8(D2),vget_high_p8(D1)),vget_high_p8(D2));
}

//len = 2: 2-Karatsuba
static inline void gfmul_2(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t D0[2], D1[2], D2[2], SAA[1], SBB[1];

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
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 3: 3-Karatsuba
static inline void gfmul_3(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t *a0, *a1, *a2, *b0, *b1, *b2, middle;
    static poly8x16_t aa01[1], bb01[1], aa02[1], bb02[1], aa12[1], bb12[1];
    static poly8x16_t D0[2], D1[2], D2[2], D3[2], D4[2], D5[2];

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
static inline void gfmul_4(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t D0[4], D1[4], D2[4], SAA[2], SBB[2];

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
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 5: 5-Karatsuba
static inline void gfmul_5(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t *a0, *b0, *a1, *b1, *a2, *b2, * a3, * b3, *a4, *b4;
    static poly8x16_t aa01[1], bb01[1], aa02[1], bb02[1], aa03[1], bb03[1],
           aa04[1], bb04[1], aa12[1], bb12[1], aa13[1], bb13[1],
           aa14[1], bb14[1], aa23[1], bb23[1], aa24[1], bb24[1], aa34[1], bb34[1];
    static poly8x16_t D0[2], D1[2], D2[2], D3[2], D4[2],
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
static inline void gfmul_6(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t D0[6], D1[6], D2[6], SAA[3], SBB[3];

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
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 8: 2-Karatsuba
static inline void gfmul_8(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t D0[8], D1[8], D2[8], SAA[4], SBB[4];

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
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 11: 3-Karatsuba
static inline void gfmul_11(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t *a0, *a1, *a2, *b0, *b1, *b2, middle;
    static poly8x16_t aa01[4], bb01[4], aa02[4], bb02[4], aa12[4], bb12[4];
    static poly8x16_t D0[8], D1[8], D2[8], D3[8], D4[8], D5[8];

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

    gfmul_4(D3, aa01, bb01);
    gfmul_4(D4, aa02, bb02);
    gfmul_4(D5, aa12, bb12);
    gfmul_4(D0, a0, b0);
    gfmul_4(D1, a1, b1);
    gfmul_3(D2, a2, b2);

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

//len = 12: 2-Karatsuba
static inline void gfmul_12(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t D0[12], D1[12], D2[12], SAA[6], SBB[6];

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
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 15: TC3_64
static inline void gfmul_15(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[10], W1[12], W4[10];
    static poly8x16_t W2[12], W3[12], tmp[12];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (poly8x16_t *)&A128[0];
    U1 = (poly8x16_t *)&A128[5];
    U2 = (poly8x16_t *)&A128[10];
    V0 = (poly8x16_t *)&B128[0];
    V1 = (poly8x16_t *)&B128[5];
    V2 = (poly8x16_t *)&B128[10];

    for (int32_t i = 0 ; i < 5 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_5(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(poly8x8_t)V1_64[0]);
    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 4; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[5] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[8],zero) ^ U2[4];
    W4[5] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[8],zero) ^ V2[4];

    for (int32_t i = 0 ; i < 5 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[5] = W0[5];
    W2[5] = W4[5];

    for (int32_t i = 0 ; i < 5 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_6(tmp, W3, W2);

    for (int32_t i = 0 ; i < 12; i++) {
        W3[i] = tmp[i];
    }

    gfmul_6(W2, W0, W4);
    gfmul_5(W4, U2, V2);
    gfmul_5(W0, U0, V0);

    for (int32_t i = 0 ; i < 12 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 10 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 9; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[9]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[18],zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[18])));
    W2[10]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[20])));
    W2[11]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[22],zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 10; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[10] = W2[10] ^ W3[10] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[16])));
    tmp[11] = W2[11] ^ W3[11] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[18], zero);
    divide_by_x_plus_one_64(W2, tmp, 24);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 9; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[9]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[18], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[18])));
    tmp[10]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[20])));
    tmp[11]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[22], zero);
    divide_by_x_plus_one_64(W3, tmp, 23);

    for (int32_t i = 0 ; i < 10 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[10] = W2[10];
    W1[11] = W2[11];

    for (int32_t i = 0 ; i < 11 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 5; i++)
    {
        int32_t j = i + 5;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 5] = W1[j] ^ W2[i];
        Out[j + 10] = W2[j] ^ W3[i];
        Out[i + 20] = W3[j] ^ W4[i];
        Out[j + 20] = W4[j];
    }

    Out[15] ^= W1[10];
    Out[16] ^= W1[11];
    Out[20] ^= W2[10];
    Out[21] ^= W2[11];
    Out[25] ^= W3[10];
}

//len = 16: 2-Karatsuba
static inline void gfmul_16(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t D0[16], D1[16], D2[16], SAA[8], SBB[8];

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
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 31: 2-Karatsuba
static inline void gfmul_31(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t D0[32], D1[32], D2[30], SAA[16], SBB[16];

    gfmul_16(D0, A, B);
    gfmul_15(D2, (A+16), (B+16));

    for(int32_t i = 0; i < 15; i++) {
        int32_t is = i + 16;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    SAA[15]=A[15];
    SBB[15]=B[15];

    gfmul_16(D1, SAA, SBB);

    for(int32_t i = 0; i < 14; i++) {
        int32_t is = i + 16;
        int32_t is2 = is + 16;
        int32_t is3 = is2 + 16;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 14; i < 16; i++) {
        int32_t is = i + 16;
        int32_t is2 = is + 16;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 32: 2-Karatsuba
static inline void gfmul_32(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t D0[32], D1[32], D2[32], SAA[16], SBB[16];

    gfmul_16(D0, A, B);
    gfmul_16(D2, (A+16), (B+16));

    for(int32_t i = 0; i < 16; i++) {
        int32_t is = i + 16;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }

    gfmul_16(D1, SAA, SBB);

    for(int32_t i = 0; i < 16; i++) {
        int32_t is = i + 16;
        int32_t is2 = is + 16;
        int32_t is3 = is2 + 16;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 33: TC3_64
static inline void gfmul_33(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[22], W1[24], W4[22];
    static poly8x16_t W2[24], W3[24], tmp[24];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (poly8x16_t *)&A128[0];
    U1 = (poly8x16_t *)&A128[11];
    U2 = (poly8x16_t *)&A128[22];
    V0 = (poly8x16_t *)&B128[0];
    V1 = (poly8x16_t *)&B128[11];
    V2 = (poly8x16_t *)&B128[22];

    for (int32_t i = 0 ; i < 11 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_11(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(poly8x8_t)V1_64[0]);
    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 10; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[11] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[20],zero) ^ U2[10];
    W4[11] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[20],zero) ^ V2[10];

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

    gfmul_12(tmp, W3, W2);

    for (int32_t i = 0 ; i < 24; i++) {
        W3[i] = tmp[i];
    }

    gfmul_12(W2, W0, W4);
    gfmul_11(W4, U2, V2);
    gfmul_11(W0, U0, V0);

    for (int32_t i = 0 ; i < 24 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 22 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[21]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[42],zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[42])));
    W2[22]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[44])));
    W2[23]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[46],zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 22; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[22] = W2[22] ^ W3[22] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[40])));
    tmp[23] = W2[23] ^ W3[23] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[42], zero);
    divide_by_x_plus_one_64(W2, tmp, 48);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[21]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[42], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[42])));
    tmp[22]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[44])));
    tmp[23]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[46], zero);
    divide_by_x_plus_one_64(W3, tmp, 47);

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

//len = 93: TC3_64
static inline void gfmul_93(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[62], W1[64], W4[62];
    static poly8x16_t W2[64], W3[64], tmp[64];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (poly8x16_t *)&A128[0];
    U1 = (poly8x16_t *)&A128[31];
    U2 = (poly8x16_t *)&A128[62];
    V0 = (poly8x16_t *)&B128[0];
    V1 = (poly8x16_t *)&B128[31];
    V2 = (poly8x16_t *)&B128[62];

    for (int32_t i = 0 ; i < 31 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_31(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(poly8x8_t)V1_64[0]);
    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 30; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[31] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[60],zero) ^ U2[30];
    W4[31] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[60],zero) ^ V2[30];

    for (int32_t i = 0 ; i < 31 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[31] = W0[31];
    W2[31] = W4[31];

    for (int32_t i = 0 ; i < 31 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_32(tmp, W3, W2);

    for (int32_t i = 0 ; i < 64; i++) {
        W3[i] = tmp[i];
    }

    gfmul_32(W2, W0, W4);
    gfmul_31(W4, U2, V2);
    gfmul_31(W0, U0, V0);

    for (int32_t i = 0 ; i < 64 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 62 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 61; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[61]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[122],zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[122])));
    W2[62]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[124])));
    W2[63]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[126],zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 62; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[62] = W2[62] ^ W3[62] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[120])));
    tmp[63] = W2[63] ^ W3[63] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[122], zero);
    divide_by_x_plus_one_64(W2, tmp, 128);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 61; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[61]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[122], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[122])));
    tmp[62]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[124])));
    tmp[63]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[126], zero);
    divide_by_x_plus_one_64(W3, tmp, 127);

    for (int32_t i = 0 ; i < 62 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[62] = W2[62];
    W1[63] = W2[63];

    for (int32_t i = 0 ; i < 63 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 31; i++)
    {
        int32_t j = i + 31;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 31] = W1[j] ^ W2[i];
        Out[j + 62] = W2[j] ^ W3[i];
        Out[i + 124] = W3[j] ^ W4[i];
        Out[j + 124] = W4[j];
    }

    Out[93] ^= W1[62];
    Out[94] ^= W1[63];
    Out[124] ^= W2[62];
    Out[125] ^= W2[63];
    Out[155] ^= W3[62];
}

//len = 94: TC3_64
static inline void gfmul_94(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t U0[32], U1[32], U2[31], V0[32], V1[32], V2[31];
    poly8x16_t W0[64], W1[64], W2[64], W3[65], W4[62];
    poly8x16_t tmp[65];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    uint64_t *A = (uint64_t *) A128;
    uint64_t *B = (uint64_t *) B128;

    for(int32_t i = 0; i < 31; i++) {
        int32_t i2 = i << 1;
        U0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2])));
        V0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2])));
        U1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + 63])));
        V1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + 63])));
        U2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + 126])));
        V2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + 126])));
    }
    U0[31]= (poly8x16_t)vcombine_p8((poly8x8_t)A[62], zero);
    V0[31]= (poly8x16_t)vcombine_p8((poly8x8_t)B[62], zero);
    U1[31]= (poly8x16_t)vcombine_p8((poly8x8_t)A[125], zero);
    V1[31]= (poly8x16_t)vcombine_p8((poly8x8_t)B[125], zero);

    for (int32_t i = 0 ; i < 31 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[31] = U0[31] ^ U1[31];
    W2[31] = V0[31] ^ V1[31];

    gfmul_32(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 31; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }

    for (int32_t i = 0 ; i < 32 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }

    for (int32_t i = 0 ; i < 32 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_32(tmp, W3, W2);

    for (int32_t i = 0 ; i < 64; i++) {
        W3[i] = tmp[i];
    }

    gfmul_32(W2, W0, W4);
    gfmul_31(W4, U2, V2);
    gfmul_32(W0, U0, V0);

    for (int32_t i = 0 ; i < 64 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 63 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 62; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[62]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[124], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[124])));
    W2[63]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[126], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 62; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[62] = W2[62] ^ W3[62] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[120])));
    tmp[63] = W2[63] ^ W3[63] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[122], zero);
    divide_by_x_plus_one_64(W2, tmp, 128);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 62; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[62]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[124], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[124])));
    tmp[63]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[126], zero);
    divide_by_x_plus_one_64(W3, tmp, 127);

    for (int32_t i = 0 ; i < 62 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[62] ^= W2[62];
    W1[63] = W2[63];

    for (int32_t i = 0 ; i < 63 ; i++) {
        W2[i] ^= W3[i];
    }

    for(int32_t i = 0; i < 62; i++) {
        Out[i] = W0[i];
        Out[i + 63] = W2[i];
        Out[i + 126] = W4[i];
    }
    Out[62] = W0[62];
    Out[125] = W2[62];
    Out[126] ^= W2[63];

    U1_64 = ((uint64_t *) &Out[31]) + 1;
    U2_64 = ((uint64_t *) &Out[94]) + 1;
    poly8x16_t aux;
    for(int32_t i = 0; i < 64; i++) {
        int32_t i2 = i << 1;
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];
        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];
        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));
    }
}

//len = 95: TC3_128
static inline void gfmul_95(poly8x16_t *Out,   poly8x16_t *A256,   poly8x16_t *B256){
    static poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[64], W1[65], W2[66], W3[66], W4[62], tmp[66];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (poly8x16_t *)&A256[0];
    U1 = (poly8x16_t *)&A256[32];
    U2 = (poly8x16_t *)&A256[64];
    V0 = (poly8x16_t *)&B256[0];
    V1 = (poly8x16_t *)&B256[32];
    V2 = (poly8x16_t *)&B256[64];

    for (int32_t i = 0 ; i < 31 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[31] = U0[31] ^ U1[31];
    W2[31] = V0[31] ^ V1[31];

    gfmul_32(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 32 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }

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

    gfmul_33(tmp, W3, W2);
    for (int32_t i = 0 ; i < 66 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_33(W2, W0, W4);

    gfmul_31(W4, U2, V2);

    gfmul_32(W0, U0, V0);

    for (int32_t i = 0 ; i < 66 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 64 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 63 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[63] = W2[64];
    W2[64] = W2[65];

    for (int32_t i = 0 ; i < 62 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 62 ; i < 65 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[65] = W3[65];
    for (int32_t i = 0 ; i < 62 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 66);

    for (int32_t i = 0 ; i < 63 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[63] = W3[64];
    tmp[64] = W3[65];
    divide_by_x_plus_one_128(tmp, W3, 65);

    for (int32_t i = 0 ; i < 62 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    for (int32_t i = 62 ; i < 64 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[64] = W2[64];

    for (int32_t i = 0 ; i < 64 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 30; i++) {
        int32_t j = i + 32;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 32] = W1[j] ^ W2[i];
        Out[j + 64] = W2[j] ^ W3[i];
        Out[i + 128] = W3[j] ^ W4[i];
        Out[j + 128] = W4[j];
    }
    for (int32_t i = 30; i < 32; i++) {
        int32_t j = i + 32;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 32] = W1[j] ^ W2[i];
        Out[j + 64] = W2[j] ^ W3[i];
        Out[i + 128] = W3[j] ^ W4[i];
    }
    Out[96] ^= W1[64];
    Out[128] ^= W2[64];
}

//len = 281: TC3_128
static inline void gfmul_281(poly8x16_t *Out,   poly8x16_t *A256,   poly8x16_t *B256){
    static poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[188], W1[189], W2[190], W3[190], W4[186], tmp[190];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (poly8x16_t *)&A256[0];
    U1 = (poly8x16_t *)&A256[94];
    U2 = (poly8x16_t *)&A256[188];
    V0 = (poly8x16_t *)&B256[0];
    V1 = (poly8x16_t *)&B256[94];
    V2 = (poly8x16_t *)&B256[188];

    for (int32_t i = 0 ; i < 93 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[93] = U0[93] ^ U1[93];
    W2[93] = V0[93] ^ V1[93];

    gfmul_94(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 94 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }

    for (int32_t i = 0 ; i < 94 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[94] = W0[94];
    W2[94] = W4[94];

    for (int32_t i = 0 ; i < 94 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_95(tmp, W3, W2);
    for (int32_t i = 0 ; i < 190 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_95(W2, W0, W4);

    gfmul_93(W4, U2, V2);

    gfmul_94(W0, U0, V0);

    for (int32_t i = 0 ; i < 190 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 188 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 187 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[187] = W2[188];
    W2[188] = W2[189];

    for (int32_t i = 0 ; i < 186 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 186 ; i < 189 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[189] = W3[189];
    for (int32_t i = 0 ; i < 186 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 190);

    for (int32_t i = 0 ; i < 187 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[187] = W3[188];
    tmp[188] = W3[189];
    divide_by_x_plus_one_128(tmp, W3, 189);

    for (int32_t i = 0 ; i < 186 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    for (int32_t i = 186 ; i < 188 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[188] = W2[188];

    for (int32_t i = 0 ; i < 188 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 92; i++) {
        int32_t j = i + 94;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 94] = W1[j] ^ W2[i];
        Out[j + 188] = W2[j] ^ W3[i];
        Out[i + 376] = W3[j] ^ W4[i];
        Out[j + 376] = W4[j];
    }
    for (int32_t i = 92; i < 94; i++) {
        int32_t j = i + 94;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 94] = W1[j] ^ W2[i];
        Out[j + 188] = W2[j] ^ W3[i];
        Out[i + 376] = W3[j] ^ W4[i];
    }
    Out[282] ^= W1[188];
    Out[376] ^= W2[188];
}


/**
 * @brief Multiply two polynomials modulo \f$ X^n - 1\f$.
 *
 * This functions multiplies a sparse polynomial <b>a1</b> (of Hamming weight equal to <b>weight</b>)
 * and a dense polynomial <b>a2</b>. The multiplication is done modulo \f$ X^n - 1\f$.
 *
 * @param[out] o Pointer to the result
 * @param[in] a1 Pointer to the sparse polynomial
 * @param[in] a2 Pointer to the dense polynomial
 * @param[in] weight Integer that is the weigt of the sparse polynomial
 * @param[in] ctx Pointer to the randomness context
 */
void vect_mul(uint64_t *o, const uint64_t *a1, const uint64_t *a2) {
    uint64_t a11[562], a22[562];
    uint64_t o_karat[1124];
    
    memcpy(a11,a1,561*8);
    memcpy(a22,a2,561*8);
    a11[561]=0;
    a22[561]=0;
    
    gfmul_281((poly8x16_t *)o_karat,   (poly8x16_t *)a11,   (poly8x16_t *)a22);
    //karatsuba(o_karat, a1, a2, VEC_N_SIZE_64, stack);
    reduce(o, o_karat);
}
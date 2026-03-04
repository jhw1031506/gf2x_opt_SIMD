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

static inline void schmul64_NEON(poly8x16_t* res, poly8x8_t* a, poly8x8_t* b){
    poly8x8_t a_shift, b_shift;
    poly8x16_t poly_tmp1, poly_tmp2, poly_res;
    poly8x8_t *p1, *p2;
    uint8x8_t mask1={0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0,0};
    uint8x8_t mask2={0xFF,0xFF,0xFF,0xFF,    0,   0,0,0};
    uint8x8_t mask3={0xFF,0xFF,   0,   0,    0,   0,0,0};
    
    poly_res=vreinterpretq_p8_p16(vmull_p8(*a,*b));
    
    a_shift=vext_p8(*a,*a,1);
    b_shift=vext_p8(*b,*b,1);
    poly_tmp1=vreinterpretq_p8_p16(vmull_p8(*a,b_shift));
    poly_tmp2=vreinterpretq_p8_p16(vmull_p8(a_shift,*b));
    poly_tmp1=vaddq_p8(poly_tmp1,poly_tmp2);
    p1=(poly8x8_t*)&poly_tmp1;
    p2=((poly8x8_t*)&poly_tmp1)+1;
    *p1=vadd_p8(*p1, *p2);
    
    *p2=vreinterpret_p8_u8(vand_u8(vreinterpret_u8_p8(*p2), mask1));
    *p1=vadd_p8(*p1, *p2);
    poly_tmp1=vextq_p8(poly_tmp1,poly_tmp1,15);

    poly_res=vaddq_p8(poly_res, poly_tmp1);

    a_shift=vext_p8(a_shift,a_shift,1);
    b_shift=vext_p8(b_shift,b_shift,1);
    poly_tmp1=vreinterpretq_p8_p16(vmull_p8(*a,b_shift));
    poly_tmp2=vreinterpretq_p8_p16(vmull_p8(a_shift,*b));
    poly_tmp1=vaddq_p8(poly_tmp1,poly_tmp2);
    *p1=vadd_p8(*p1, *p2);
    *p2=vreinterpret_p8_u8(vand_u8(vreinterpret_u8_p8(*p2), mask2));
    *p1=vadd_p8(*p1, *p2);
    poly_tmp1=vextq_p8(poly_tmp1,poly_tmp1,14);
    
    poly_res=vaddq_p8(poly_res, poly_tmp1);
    
    a_shift=vext_p8(a_shift,a_shift,1);
    b_shift=vext_p8(b_shift,b_shift,1);
    
    poly_tmp1=vreinterpretq_p8_p16(vmull_p8(*a,b_shift));
    poly_tmp2=vreinterpretq_p8_p16(vmull_p8(a_shift,*b));
    poly_tmp1=vaddq_p8(poly_tmp1,poly_tmp2);
    *p1=vadd_p8(*p1, *p2);
    *p2=vreinterpret_p8_u8(vand_u8(vreinterpret_u8_p8(*p2), mask3));
    *p1=vadd_p8(*p1, *p2);
    poly_tmp1=vextq_p8(poly_tmp1,poly_tmp1,13);
    
    poly_res=vaddq_p8(poly_res, poly_tmp1);

    b_shift=vext_p8(b_shift,b_shift,1);
    poly_tmp1=vreinterpretq_p8_p16(vmull_p8(*a,b_shift));
    *p1=vadd_p8(*p1, *p2);
    *p2=vadd_p8(*p2, *p2);
    poly_tmp1=vextq_p8(poly_tmp1,poly_tmp1,12);
    
    *res = vaddq_p8(poly_res, poly_tmp1);
}
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
    schmul64_NEON(&D1, &alpah, &blpbh);
    schmul64_NEON(&D0, &al, &bl);
    schmul64_NEON(&D2, &ah, &bh);

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

//len = 7: 2-Karatsuba
static inline void gfmul_7(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){
    static poly8x16_t D0[8], D1[8], D2[6], SAA[4], SBB[4];

    gfmul_4(D0, A, B);
    gfmul_3(D2, (A+4), (B+4));

    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 4;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    SAA[3]=A[3];
    SBB[3]=B[3];

    gfmul_4(D1, SAA, SBB);

    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 4;
        int32_t is2 = is + 4;
        int32_t is3 = is2 + 4;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 2; i < 4; i++) {
        int32_t is = i + 4;
        int32_t is2 = is + 4;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
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

//len = 17: TC3_128
static inline void gfmul_17(poly8x16_t *Out,   poly8x16_t *A256,   poly8x16_t *B256){
    static poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[12], W1[13], W2[14], W3[14], W4[10], tmp[14];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (poly8x16_t *)&A256[0];
    U1 = (poly8x16_t *)&A256[6];
    U2 = (poly8x16_t *)&A256[12];
    V0 = (poly8x16_t *)&B256[0];
    V1 = (poly8x16_t *)&B256[6];
    V2 = (poly8x16_t *)&B256[12];

    for (int32_t i = 0 ; i < 5 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[5] = U0[5] ^ U1[5];
    W2[5] = V0[5] ^ V1[5];

    gfmul_6(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 6 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }

    for (int32_t i = 0 ; i < 6 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[6] = W0[6];
    W2[6] = W4[6];

    for (int32_t i = 0 ; i < 6 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_7(tmp, W3, W2);
    for (int32_t i = 0 ; i < 14 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_7(W2, W0, W4);

    gfmul_5(W4, U2, V2);

    gfmul_6(W0, U0, V0);

    for (int32_t i = 0 ; i < 14 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 12 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 11 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[11] = W2[12];
    W2[12] = W2[13];

    for (int32_t i = 0 ; i < 10 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 10 ; i < 13 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[13] = W3[13];
    for (int32_t i = 0 ; i < 10 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 14);

    for (int32_t i = 0 ; i < 11 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[11] = W3[12];
    tmp[12] = W3[13];
    divide_by_x_plus_one_128(tmp, W3, 13);

    for (int32_t i = 0 ; i < 10 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    for (int32_t i = 10 ; i < 12 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[12] = W2[12];

    for (int32_t i = 0 ; i < 12 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 4; i++) {
        int32_t j = i + 6;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 6] = W1[j] ^ W2[i];
        Out[j + 12] = W2[j] ^ W3[i];
        Out[i + 24] = W3[j] ^ W4[i];
        Out[j + 24] = W4[j];
    }
    for (int32_t i = 4; i < 6; i++) {
        int32_t j = i + 6;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 6] = W1[j] ^ W2[i];
        Out[j + 12] = W2[j] ^ W3[i];
        Out[i + 24] = W3[j] ^ W4[i];
    }
    Out[18] ^= W1[12];
    Out[24] ^= W2[12];
}

//len = 18: TC3_64
static inline void gfmul_18(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[12], W1[14], W4[12];
    static poly8x16_t W2[14], W3[14], tmp[14];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (poly8x16_t *)&A128[0];
    U1 = (poly8x16_t *)&A128[6];
    U2 = (poly8x16_t *)&A128[12];
    V0 = (poly8x16_t *)&B128[0];
    V1 = (poly8x16_t *)&B128[6];
    V2 = (poly8x16_t *)&B128[12];

    for (int32_t i = 0 ; i < 6 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_6(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(poly8x8_t)V1_64[0]);
    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 5; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[6] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[10],zero) ^ U2[5];
    W4[6] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[10],zero) ^ V2[5];

    for (int32_t i = 0 ; i < 6 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[6] = W0[6];
    W2[6] = W4[6];

    for (int32_t i = 0 ; i < 6 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_7(tmp, W3, W2);

    for (int32_t i = 0 ; i < 14; i++) {
        W3[i] = tmp[i];
    }

    gfmul_7(W2, W0, W4);
    gfmul_6(W4, U2, V2);
    gfmul_6(W0, U0, V0);

    for (int32_t i = 0 ; i < 14 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 12 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 11; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[11]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[22],zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[22])));
    W2[12]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[24])));
    W2[13]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[26],zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 12; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[12] = W2[12] ^ W3[12] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[20])));
    tmp[13] = W2[13] ^ W3[13] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[22], zero);
    divide_by_x_plus_one_64(W2, tmp, 28);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 11; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[11]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[22], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[22])));
    tmp[12]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[24])));
    tmp[13]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[26], zero);
    divide_by_x_plus_one_64(W3, tmp, 27);

    for (int32_t i = 0 ; i < 12 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[12] = W2[12];
    W1[13] = W2[13];

    for (int32_t i = 0 ; i < 13 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 6; i++)
    {
        int32_t j = i + 6;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 6] = W1[j] ^ W2[i];
        Out[j + 12] = W2[j] ^ W3[i];
        Out[i + 24] = W3[j] ^ W4[i];
        Out[j + 24] = W4[j];
    }

    Out[18] ^= W1[12];
    Out[19] ^= W1[13];
    Out[24] ^= W2[12];
    Out[25] ^= W2[13];
    Out[30] ^= W3[12];
}

//len = 49: TC3_64
static inline void gfmul_49(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t U0[17], U1[17], U2[16], V0[17], V1[17], V2[16];
    poly8x16_t W0[34], W1[34], W2[34], W3[35], W4[32];
    poly8x16_t tmp[35];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    uint64_t *A = (uint64_t *) A128;
    uint64_t *B = (uint64_t *) B128;

    for(int32_t i = 0; i < 16; i++) {
        int32_t i2 = i << 1;
        U0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2])));
        V0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2])));
        U1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + 33])));
        V1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + 33])));
        U2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + 66])));
        V2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + 66])));
    }
    U0[16]= (poly8x16_t)vcombine_p8((poly8x8_t)A[32], zero);
    V0[16]= (poly8x16_t)vcombine_p8((poly8x8_t)B[32], zero);
    U1[16]= (poly8x16_t)vcombine_p8((poly8x8_t)A[65], zero);
    V1[16]= (poly8x16_t)vcombine_p8((poly8x8_t)B[65], zero);

    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[16] = U0[16] ^ U1[16];
    W2[16] = V0[16] ^ V1[16];

    gfmul_17(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 16; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
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

    gfmul_17(tmp, W3, W2);

    for (int32_t i = 0 ; i < 34; i++) {
        W3[i] = tmp[i];
    }

    gfmul_17(W2, W0, W4);
    gfmul_16(W4, U2, V2);
    gfmul_17(W0, U0, V0);

    for (int32_t i = 0 ; i < 34 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 33 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 32; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[32]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[64], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[64])));
    W2[33]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[66], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 32; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[32] = W2[32] ^ W3[32] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[60])));
    tmp[33] = W2[33] ^ W3[33] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[62], zero);
    divide_by_x_plus_one_64(W2, tmp, 68);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 32; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[32]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[64], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[64])));
    tmp[33]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[66], zero);
    divide_by_x_plus_one_64(W3, tmp, 67);

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

    U1_64 = ((uint64_t *) &Out[16]) + 1;
    U2_64 = ((uint64_t *) &Out[49]) + 1;
    poly8x16_t aux;
    for(int32_t i = 0; i < 34; i++) {
        int32_t i2 = i << 1;
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];
        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];
        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));
    }
}

//len = 50: TC3_128
static inline void gfmul_50(poly8x16_t *Out,   poly8x16_t *A256,   poly8x16_t *B256){
    static poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[34], W1[35], W2[36], W3[36], W4[32], tmp[36];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (poly8x16_t *)&A256[0];
    U1 = (poly8x16_t *)&A256[17];
    U2 = (poly8x16_t *)&A256[34];
    V0 = (poly8x16_t *)&B256[0];
    V1 = (poly8x16_t *)&B256[17];
    V2 = (poly8x16_t *)&B256[34];

    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[16] = U0[16] ^ U1[16];
    W2[16] = V0[16] ^ V1[16];

    gfmul_17(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 17 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }

    for (int32_t i = 0 ; i < 17 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[17] = W0[17];
    W2[17] = W4[17];

    for (int32_t i = 0 ; i < 17 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_18(tmp, W3, W2);
    for (int32_t i = 0 ; i < 36 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_18(W2, W0, W4);

    gfmul_16(W4, U2, V2);

    gfmul_17(W0, U0, V0);

    for (int32_t i = 0 ; i < 36 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 34 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 33 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[33] = W2[34];
    W2[34] = W2[35];

    for (int32_t i = 0 ; i < 32 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 32 ; i < 35 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[35] = W3[35];
    for (int32_t i = 0 ; i < 32 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 36);

    for (int32_t i = 0 ; i < 33 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[33] = W3[34];
    tmp[34] = W3[35];
    divide_by_x_plus_one_128(tmp, W3, 35);

    for (int32_t i = 0 ; i < 32 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    for (int32_t i = 32 ; i < 34 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[34] = W2[34];

    for (int32_t i = 0 ; i < 34 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 15; i++) {
        int32_t j = i + 17;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 17] = W1[j] ^ W2[i];
        Out[j + 34] = W2[j] ^ W3[i];
        Out[i + 68] = W3[j] ^ W4[i];
        Out[j + 68] = W4[j];
    }
    for (int32_t i = 15; i < 17; i++) {
        int32_t j = i + 17;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 17] = W1[j] ^ W2[i];
        Out[j + 34] = W2[j] ^ W3[i];
        Out[i + 68] = W3[j] ^ W4[i];
    }
    Out[51] ^= W1[34];
    Out[68] ^= W2[34];
}

//len = 51: TC3_64
static inline void gfmul_51(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[34], W1[36], W4[34];
    static poly8x16_t W2[36], W3[36], tmp[36];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (poly8x16_t *)&A128[0];
    U1 = (poly8x16_t *)&A128[17];
    U2 = (poly8x16_t *)&A128[34];
    V0 = (poly8x16_t *)&B128[0];
    V1 = (poly8x16_t *)&B128[17];
    V2 = (poly8x16_t *)&B128[34];

    for (int32_t i = 0 ; i < 17 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_17(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(poly8x8_t)V1_64[0]);
    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 16; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[17] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[32],zero) ^ U2[16];
    W4[17] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[32],zero) ^ V2[16];

    for (int32_t i = 0 ; i < 17 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[17] = W0[17];
    W2[17] = W4[17];

    for (int32_t i = 0 ; i < 17 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_18(tmp, W3, W2);

    for (int32_t i = 0 ; i < 36; i++) {
        W3[i] = tmp[i];
    }

    gfmul_18(W2, W0, W4);
    gfmul_17(W4, U2, V2);
    gfmul_17(W0, U0, V0);

    for (int32_t i = 0 ; i < 36 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 34 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 33; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[33]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[66],zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[66])));
    W2[34]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[68])));
    W2[35]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[70],zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 34; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[34] = W2[34] ^ W3[34] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[64])));
    tmp[35] = W2[35] ^ W3[35] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[66], zero);
    divide_by_x_plus_one_64(W2, tmp, 72);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 33; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[33]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[66], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[66])));
    tmp[34]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[68])));
    tmp[35]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[70], zero);
    divide_by_x_plus_one_64(W3, tmp, 71);

    for (int32_t i = 0 ; i < 34 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[34] = W2[34];
    W1[35] = W2[35];

    for (int32_t i = 0 ; i < 35 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 17; i++)
    {
        int32_t j = i + 17;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 17] = W1[j] ^ W2[i];
        Out[j + 34] = W2[j] ^ W3[i];
        Out[i + 68] = W3[j] ^ W4[i];
        Out[j + 68] = W4[j];
    }

    Out[51] ^= W1[34];
    Out[52] ^= W1[35];
    Out[68] ^= W2[34];
    Out[69] ^= W2[35];
    Out[85] ^= W3[34];
}

//len = 52: TC3_64
static inline void gfmul_52(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t U0[18], U1[18], U2[17], V0[18], V1[18], V2[17];
    poly8x16_t W0[36], W1[36], W2[36], W3[37], W4[34];
    poly8x16_t tmp[37];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    uint64_t *A = (uint64_t *) A128;
    uint64_t *B = (uint64_t *) B128;

    for(int32_t i = 0; i < 17; i++) {
        int32_t i2 = i << 1;
        U0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2])));
        V0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2])));
        U1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + 35])));
        V1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + 35])));
        U2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + 70])));
        V2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + 70])));
    }
    U0[17]= (poly8x16_t)vcombine_p8((poly8x8_t)A[34], zero);
    V0[17]= (poly8x16_t)vcombine_p8((poly8x8_t)B[34], zero);
    U1[17]= (poly8x16_t)vcombine_p8((poly8x8_t)A[69], zero);
    V1[17]= (poly8x16_t)vcombine_p8((poly8x8_t)B[69], zero);

    for (int32_t i = 0 ; i < 17 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[17] = U0[17] ^ U1[17];
    W2[17] = V0[17] ^ V1[17];

    gfmul_18(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 17; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
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

    gfmul_18(tmp, W3, W2);

    for (int32_t i = 0 ; i < 36; i++) {
        W3[i] = tmp[i];
    }

    gfmul_18(W2, W0, W4);
    gfmul_17(W4, U2, V2);
    gfmul_18(W0, U0, V0);

    for (int32_t i = 0 ; i < 36 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 35 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 34; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[34]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[68], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[68])));
    W2[35]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[70], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 34; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[34] = W2[34] ^ W3[34] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[64])));
    tmp[35] = W2[35] ^ W3[35] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[66], zero);
    divide_by_x_plus_one_64(W2, tmp, 72);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 34; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[34]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[68], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[68])));
    tmp[35]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[70], zero);
    divide_by_x_plus_one_64(W3, tmp, 71);

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

    U1_64 = ((uint64_t *) &Out[17]) + 1;
    U2_64 = ((uint64_t *) &Out[52]) + 1;
    poly8x16_t aux;
    for(int32_t i = 0; i < 36; i++) {
        int32_t i2 = i << 1;
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];
        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];
        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));
    }
}

//len = 149: TC3_64
static inline void gfmul_149(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[100], W1[101], W4[98];
    static poly8x16_t W2[102], W3[101], tmp[102];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (poly8x16_t *)&A128[0];
    U1 = (poly8x16_t *)&A128[50];
    U2 = (poly8x16_t *)&A128[100];
    V0 = (poly8x16_t *)&B128[0];
    V1 = (poly8x16_t *)&B128[50];
    V2 = (poly8x16_t *)&B128[100];

    for (int32_t i = 0 ; i < 49 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    W3[49] = U0[49] ^ U1[49];
    W2[49] = V0[49] ^ V1[49];

    gfmul_50(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 49; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }

    W0[50] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[98], zero);
    W4[50] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[98], zero);

    for (int32_t i = 0 ; i < 50 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[50] = W0[50];
    W2[50] = W4[50];

    for (int32_t i = 0 ; i < 50 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_51(tmp, W3, W2);

    for (int32_t i = 0 ; i < 101; i++) {
        W3[i] = tmp[i];
    }

    gfmul_51(W2, W0, W4);
    gfmul_49(W4, U2, V2);
    gfmul_50(W0, U0, V0);

    for (int32_t i = 0 ; i < 101 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 100 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;

    for(int32_t i = 0; i < 99; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[99]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[198]))) ^ (poly8x16_t)vcombine_p8((poly8x8_t)U2_64[198], zero);
    W2[100]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[200], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 98; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[98] = W2[98] ^ W3[98] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[192])));
    tmp[99] = W2[99] ^ W3[99] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[194], zero);
    tmp[100] = W2[100] ^ W3[100];
    divide_by_x_plus_one_64(W2, tmp, 202);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 99; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }

    tmp[99]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[198]))) ^ (poly8x16_t)vcombine_p8((poly8x8_t)U2_64[198], zero);
    tmp[100]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[200], zero);
    divide_by_x_plus_one_64(W3, tmp, 201);

    for (int32_t i = 0 ; i < 98 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[98] ^= W2[98];
    W1[99] ^= W2[99];
    W1[100] = W2[100];

    for (int32_t i = 0 ; i < 100 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 48; i++)
    {
        int32_t j = i + 50;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 50] = W1[j] ^ W2[i];
        Out[j + 100] = W2[j] ^ W3[i];
        Out[i + 200] = W3[j] ^ W4[i];
        Out[j + 200] = W4[j];
    }

    for (int32_t i = 48; i < 50; i++)
    {
        int32_t j = i + 50;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 50] = W1[j] ^ W2[i];
        Out[j + 100] = W2[j] ^ W3[i];
        Out[i + 200] = W3[j] ^ W4[i];
    }
    Out[150] ^= W1[100];
    Out[200] ^= W2[100];
}

//len = 151: TC3_64
static inline void gfmul_151(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){
    poly8x16_t U0[51], U1[51], U2[50], V0[51], V1[51], V2[50];
    poly8x16_t W0[102], W1[102], W2[102], W3[103], W4[100];
    poly8x16_t tmp[103];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    uint64_t *A = (uint64_t *) A128;
    uint64_t *B = (uint64_t *) B128;

    for(int32_t i = 0; i < 50; i++) {
        int32_t i2 = i << 1;
        U0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2])));
        V0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2])));
        U1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + 101])));
        V1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + 101])));
        U2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + 202])));
        V2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + 202])));
    }
    U0[50]= (poly8x16_t)vcombine_p8((poly8x8_t)A[100], zero);
    V0[50]= (poly8x16_t)vcombine_p8((poly8x8_t)B[100], zero);
    U1[50]= (poly8x16_t)vcombine_p8((poly8x8_t)A[201], zero);
    V1[50]= (poly8x16_t)vcombine_p8((poly8x8_t)B[201], zero);

    for (int32_t i = 0 ; i < 50 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[50] = U0[50] ^ U1[50];
    W2[50] = V0[50] ^ V1[50];

    gfmul_51(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 50; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }

    for (int32_t i = 0 ; i < 51 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }

    for (int32_t i = 0 ; i < 51 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_51(tmp, W3, W2);

    for (int32_t i = 0 ; i < 102; i++) {
        W3[i] = tmp[i];
    }

    gfmul_51(W2, W0, W4);
    gfmul_50(W4, U2, V2);
    gfmul_51(W0, U0, V0);

    for (int32_t i = 0 ; i < 102 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 101 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 100; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[100]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[200], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[200])));
    W2[101]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[202], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 100; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[100] = W2[100] ^ W3[100] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[196])));
    tmp[101] = W2[101] ^ W3[101] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[198], zero);
    divide_by_x_plus_one_64(W2, tmp, 204);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 100; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[100]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[200], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[200])));
    tmp[101]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[202], zero);
    divide_by_x_plus_one_64(W3, tmp, 203);

    for (int32_t i = 0 ; i < 100 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[100] ^= W2[100];
    W1[101] = W2[101];

    for (int32_t i = 0 ; i < 101 ; i++) {
        W2[i] ^= W3[i];
    }

    for(int32_t i = 0; i < 100; i++) {
        Out[i] = W0[i];
        Out[i + 101] = W2[i];
        Out[i + 202] = W4[i];
    }
    Out[100] = W0[100];
    Out[201] = W2[100];
    Out[202] ^= W2[101];

    U1_64 = ((uint64_t *) &Out[50]) + 1;
    U2_64 = ((uint64_t *) &Out[151]) + 1;
    poly8x16_t aux;
    for(int32_t i = 0; i < 102; i++) {
        int32_t i2 = i << 1;
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];
        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];
        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));
    }
}

//len = 152: TC3_128
static inline void gfmul_152(poly8x16_t *Out,   poly8x16_t *A256,   poly8x16_t *B256){
    static poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[102], W1[103], W2[104], W3[104], W4[100], tmp[104];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (poly8x16_t *)&A256[0];
    U1 = (poly8x16_t *)&A256[51];
    U2 = (poly8x16_t *)&A256[102];
    V0 = (poly8x16_t *)&B256[0];
    V1 = (poly8x16_t *)&B256[51];
    V2 = (poly8x16_t *)&B256[102];

    for (int32_t i = 0 ; i < 50 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[50] = U0[50] ^ U1[50];
    W2[50] = V0[50] ^ V1[50];

    gfmul_51(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 51 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }

    for (int32_t i = 0 ; i < 51 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[51] = W0[51];
    W2[51] = W4[51];

    for (int32_t i = 0 ; i < 51 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_52(tmp, W3, W2);
    for (int32_t i = 0 ; i < 104 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_52(W2, W0, W4);

    gfmul_50(W4, U2, V2);

    gfmul_51(W0, U0, V0);

    for (int32_t i = 0 ; i < 104 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 102 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 101 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[101] = W2[102];
    W2[102] = W2[103];

    for (int32_t i = 0 ; i < 100 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 100 ; i < 103 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[103] = W3[103];
    for (int32_t i = 0 ; i < 100 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 104);

    for (int32_t i = 0 ; i < 101 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[101] = W3[102];
    tmp[102] = W3[103];
    divide_by_x_plus_one_128(tmp, W3, 103);

    for (int32_t i = 0 ; i < 100 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    for (int32_t i = 100 ; i < 102 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[102] = W2[102];

    for (int32_t i = 0 ; i < 102 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 49; i++) {
        int32_t j = i + 51;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 51] = W1[j] ^ W2[i];
        Out[j + 102] = W2[j] ^ W3[i];
        Out[i + 204] = W3[j] ^ W4[i];
        Out[j + 204] = W4[j];
    }
    for (int32_t i = 49; i < 51; i++) {
        int32_t j = i + 51;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 51] = W1[j] ^ W2[i];
        Out[j + 102] = W2[j] ^ W3[i];
        Out[i + 204] = W3[j] ^ W4[i];
    }
    Out[153] ^= W1[102];
    Out[204] ^= W2[102];
}

//len = 451: TC3_128
static inline void gfmul_451(poly8x16_t *Out,   poly8x16_t *A256,   poly8x16_t *B256){
    static poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[302], W1[303], W2[304], W3[304], W4[298], tmp[304];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (poly8x16_t *)&A256[0];
    U1 = (poly8x16_t *)&A256[151];
    U2 = (poly8x16_t *)&A256[302];
    V0 = (poly8x16_t *)&B256[0];
    V1 = (poly8x16_t *)&B256[151];
    V2 = (poly8x16_t *)&B256[302];

    for (int32_t i = 0 ; i < 149 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[149] = U0[149] ^ U1[149];
    W2[149] = V0[149] ^ V1[149];
    W3[150] = U0[150] ^ U1[150];
    W2[150] = V0[150] ^ V1[150];

    gfmul_151(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 150 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    W0[151] = U1[150];
    W4[151] = V1[150];

    for (int32_t i = 0 ; i < 151 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[151] = W0[151];
    W2[151] = W4[151];

    for (int32_t i = 0 ; i < 151 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_152(tmp, W3, W2);
    for (int32_t i = 0 ; i < 304 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_152(W2, W0, W4);

    gfmul_149(W4, U2, V2);

    gfmul_151(W0, U0, V0);

    for (int32_t i = 0 ; i < 304 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 302 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 301 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[301] = W2[302];
    W2[302] = W2[303];

    for (int32_t i = 0 ; i < 298 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 298 ; i < 303 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[303] = W3[303];
    for (int32_t i = 0 ; i < 298 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 304);

    for (int32_t i = 0 ; i < 301 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[301] = W3[302];
    tmp[302] = W3[303];
    divide_by_x_plus_one_128(tmp, W3, 303);

    for (int32_t i = 0 ; i < 298 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 298 ; i < 302 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[302] = W2[302];

    for (int32_t i = 0 ; i < 302 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 147; i++) {
        int32_t j = i + 151;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 151] = W1[j] ^ W2[i];
        Out[j + 302] = W2[j] ^ W3[i];
        Out[i + 604] = W3[j] ^ W4[i];
        Out[j + 604] = W4[j];
    }
    for (int32_t i = 147; i < 151; i++) {
        int32_t j = i + 151;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 151] = W1[j] ^ W2[i];
        Out[j + 302] = W2[j] ^ W3[i];
        Out[i + 604] = W3[j] ^ W4[i];
    }
    Out[453] ^= W1[302];
    Out[604] ^= W2[302];
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
    uint64_t a11[902], a22[902];
    uint64_t o_karat[1804];
    
    memcpy(a11,a1,901*8);
    memcpy(a22,a2,901*8);
    a11[901]=0;
    a22[901]=0;

    
    gfmul_451((poly8x16_t *)o_karat,   (poly8x16_t *)a11,   (poly8x16_t *)a22);
    //karatsuba(o_karat, a1, a2, VEC_N_SIZE_64, stack);
    reduce(o, o_karat);
}
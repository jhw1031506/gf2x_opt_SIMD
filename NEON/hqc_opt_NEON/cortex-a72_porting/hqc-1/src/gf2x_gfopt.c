#include "gf2x.h"
#include "parameters.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <arm_neon.h>

#define N_MOD_64   (5)

 /**
  * @brief Compute o(x) = a(x) mod \f$ X^n - 1\f$
  *
  * This function computes the modular reduction of the polynomial a(x)
  *
  * @param[out] o Pointer to the result
  * @param[in] a Pointer to the polynomial a(x)
  */
void reduce(uint64_t *o, const uint64_t *a) {

    for(int i=0;i<VEC_N_SIZE_64;i++) {
        o[i] = a[i] ^ ((a[VEC_N_SIZE_64-1+i]>>N_MOD_64) | (a[VEC_N_SIZE_64+i]<<(64-N_MOD_64)));
    }
    o[VEC_N_SIZE_64-1] &= BITMASK(PARAM_N, 64);
}

//========================================================================
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
 
//len = 1: SB_NEON
static inline void gfmul_1(poly8x16_t* Out, const poly8x16_t* a, const poly8x16_t* b){;
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
static inline void gfmul_2(poly8x16_t *Out,  const poly8x16_t *A,  const poly8x16_t *B){
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
static inline void gfmul_3(poly8x16_t *Out,  const poly8x16_t *A,  const poly8x16_t *B){
    const poly8x16_t *a0, *a1, *a2, *b0, *b1, *b2;
    static poly8x16_t aa01[1], bb01[1], aa02[1], bb02[1], aa12[1], bb12[1];
    static poly8x16_t D0[2], D1[2], D2[2], D3[2], D4[2], D5[2], middle;

    a0 = A;
    a1 = A + 1;
    a2 = A + 2;
    b0 = B;
    b1 = B + 1;
    b2 = B + 2;

    aa01[0] = a0[0] ^ a1[0];
    bb01[0] = b0[0] ^ b1[0];
    aa12[0] = a2[0] ^ a1[0];
    bb12[0] = b2[0] ^ b1[0];
    aa02[0] = a0[0] ^ a2[0];
    bb02[0] = b0[0] ^ b2[0];
    

    gfmul_1(D3, aa01, bb01);
    gfmul_1(D4, aa02, bb02);
    gfmul_1(D5, aa12, bb12);
    gfmul_1(D0, a0, b0);
    gfmul_1(D1, a1, b1);
    gfmul_1(D2, a2, b2);

    middle = D0[0] ^ D1[0] ^ D0[1];
    Out[0] = D0[0];
    Out[1] = D3[0] ^ middle;
    Out[2] = D4[0] ^ D2[0] ^ D3[1] ^ D1[1] ^ middle;
    middle = D1[1] ^ D2[0] ^ D2[1];
    Out[3] = D5[0] ^ D4[1] ^ D0[1] ^ D1[0] ^ middle;
    Out[4] = D5[1] ^ middle;
    Out[5] = D2[1];

}

//len = 4: 2-Karatsuba
static inline void gfmul_4(poly8x16_t *Out,  const poly8x16_t *A,  const poly8x16_t *B){
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

//len = 7: 2-Karatsuba
static inline void gfmul_7(poly8x16_t *Out,  const poly8x16_t *A,  const poly8x16_t *B){
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
static inline void gfmul_8(poly8x16_t *Out,  const poly8x16_t *A,  const poly8x16_t *B){
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

//len = 9: 3-Karatsuba
static inline void gfmul_9(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
    const poly8x16_t *a0, *a1, *a2, *b0, *b1, *b2;
    static poly8x16_t aa01[3], bb01[3], aa02[3], bb02[3], aa12[3], bb12[3], middle;
    static poly8x16_t D0[6], D1[6], D2[6], D3[6], D4[6], D5[6];

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

//len = 15: 2-Karatsuba
static inline void gfmul_15(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
    static poly8x16_t D0[16], D1[16], D2[14], SAA[8], SBB[8];

    gfmul_8(D0, A, B);
    gfmul_7(D2, (A+8), (B+8));

    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 8;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    SAA[7]=A[7];
    SBB[7]=B[7];

    gfmul_8(D1, SAA, SBB);

    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 8;
        int32_t is2 = is + 8;
        int32_t is3 = is2 + 8;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 6; i < 8; i++) {
        int32_t is = i + 8;
        int32_t is2 = is + 8;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 16: 2-Karatsuba
static inline void gfmul_16(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
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

//len = 17: 2-Karatsuba
static inline void gfmul_17(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
    static poly8x16_t D0[18], D1[18], D2[16], SAA[9], SBB[9];

    gfmul_9(D0, A, B);
    gfmul_8(D2, (A+9), (B+9));

    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 9;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    SAA[8]=A[8];
    SBB[8]=B[8];

    gfmul_9(D1, SAA, SBB);

    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 9;
        int32_t is2 = is + 9;
        int32_t is3 = is2 + 9;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 7; i < 9; i++) {
        int32_t is = i + 9;
        int32_t is2 = is + 9;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 45: TC3_64
static inline void gfmul_45(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[30], W1[32], W4[30];
    static poly8x16_t W2[32], W3[32], tmp[32];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[15];
    U2 = (const poly8x16_t *)&A128[30];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[15];
    V2 = (const poly8x16_t *)&B128[30];

    for (int32_t i = 0 ; i < 15 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_15(W1, W2, W3);

    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(poly8x8_t)V1_64[0]);
    U1_64 = ((const uint64_t *) U1) + 1;
    V1_64 = ((const uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 14; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[15] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[28],zero) ^ U2[14];
    W4[15] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[28],zero) ^ V2[14];

    for (int32_t i = 0 ; i < 15 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[15] = W0[15];
    W2[15] = W4[15];

    for (int32_t i = 0 ; i < 15 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_16(tmp, W3, W2);

    for (int32_t i = 0 ; i < 32; i++) {
        W3[i] = tmp[i];
    }

    gfmul_16(W2, W0, W4);
    gfmul_15(W4, U2, V2);
    gfmul_15(W0, U0, V0);

    for (int32_t i = 0 ; i < 32 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 30 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 29; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[29]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[58],zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[58])));
    W2[30]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[60])));
    W2[31]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[62],zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 30; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[30] = W2[30] ^ W3[30] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[56])));
    tmp[31] = W2[31] ^ W3[31] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[58], zero);
    divide_by_x_plus_one_64(W2, tmp, 64);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 29; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[29]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[58], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[58])));
    tmp[30]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[60])));
    tmp[31]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[62], zero);
    divide_by_x_plus_one_64(W3, tmp, 63);

    for (int32_t i = 0 ; i < 30 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[30] = W2[30];
    W1[31] = W2[31];

    for (int32_t i = 0 ; i < 31 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 15; i++)
    {
        int32_t j = i + 15;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 15] = W1[j] ^ W2[i];
        Out[j + 30] = W2[j] ^ W3[i];
        Out[i + 60] = W3[j] ^ W4[i];
        Out[j + 60] = W4[j];
    }

    Out[45] ^= W1[30];
    Out[46] ^= W1[31];
    Out[60] ^= W2[30];
    Out[61] ^= W2[31];
    Out[75] ^= W3[30];
}

//len = 47: TC3_128
static inline void gfmul_47(poly8x16_t *Out,  const  poly8x16_t *A256,  const  poly8x16_t *B256){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[32], W1[33], W2[34], W3[34], W4[30], tmp[34];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (const poly8x16_t *)&A256[0];
    U1 = (const poly8x16_t *)&A256[16];
    U2 = (const poly8x16_t *)&A256[32];
    V0 = (const poly8x16_t *)&B256[0];
    V1 = (const poly8x16_t *)&B256[16];
    V2 = (const poly8x16_t *)&B256[32];

    for (int32_t i = 0 ; i < 15 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[15] = U0[15] ^ U1[15];
    W2[15] = V0[15] ^ V1[15];

    gfmul_16(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 16 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }

    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[16] = W0[16];
    W2[16] = W4[16];

    for (int32_t i = 0 ; i < 16 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_17(tmp, W3, W2);
    for (int32_t i = 0 ; i < 34 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_17(W2, W0, W4);

    gfmul_15(W4, U2, V2);

    gfmul_16(W0, U0, V0);

    for (int32_t i = 0 ; i < 34 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 32 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 31 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[31] = W2[32];
    W2[32] = W2[33];

    for (int32_t i = 0 ; i < 30 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 30 ; i < 33 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[33] = W3[33];
    for (int32_t i = 0 ; i < 30 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 34);

    for (int32_t i = 0 ; i < 31 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[31] = W3[32];
    tmp[32] = W3[33];
    divide_by_x_plus_one_128(tmp, W3, 33);

    for (int32_t i = 0 ; i < 30 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    for (int32_t i = 30 ; i < 32 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[32] = W2[32];

    for (int32_t i = 0 ; i < 32 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 14; i++) {
        int32_t j = i + 16;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 16] = W1[j] ^ W2[i];
        Out[j + 32] = W2[j] ^ W3[i];
        Out[i + 64] = W3[j] ^ W4[i];
        Out[j + 64] = W4[j];
    }
    for (int32_t i = 14; i < 16; i++) {
        int32_t j = i + 16;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 16] = W1[j] ^ W2[i];
        Out[j + 32] = W2[j] ^ W3[i];
        Out[i + 64] = W3[j] ^ W4[i];
    }
    Out[48] ^= W1[32];
    Out[64] ^= W2[32];
}

//len = 48: TC3_64
static inline void gfmul_48(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[32], W1[34], W4[32];
    static poly8x16_t W2[34], W3[34], tmp[34];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[16];
    U2 = (const poly8x16_t *)&A128[32];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[16];
    V2 = (const poly8x16_t *)&B128[32];

    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_16(W1, W2, W3);

    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(poly8x8_t)V1_64[0]);
    U1_64 = ((const uint64_t *) U1) + 1;
    V1_64 = ((const uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 15; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[16] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[30],zero) ^ U2[15];
    W4[16] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[30],zero) ^ V2[15];

    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[16] = W0[16];
    W2[16] = W4[16];

    for (int32_t i = 0 ; i < 16 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_17(tmp, W3, W2);

    for (int32_t i = 0 ; i < 34; i++) {
        W3[i] = tmp[i];
    }

    gfmul_17(W2, W0, W4);
    gfmul_16(W4, U2, V2);
    gfmul_16(W0, U0, V0);

    for (int32_t i = 0 ; i < 34 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 32 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 31; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[31]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[62],zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[62])));
    W2[32]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[64])));
    W2[33]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[66],zero);

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
    for(int32_t i = 0; i < 31; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[31]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[62], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[62])));
    tmp[32]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[64])));
    tmp[33]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[66], zero);
    divide_by_x_plus_one_64(W3, tmp, 67);

    for (int32_t i = 0 ; i < 32 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[32] = W2[32];
    W1[33] = W2[33];

    for (int32_t i = 0 ; i < 33 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 16; i++)
    {
        int32_t j = i + 16;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 16] = W1[j] ^ W2[i];
        Out[j + 32] = W2[j] ^ W3[i];
        Out[i + 64] = W3[j] ^ W4[i];
        Out[j + 64] = W4[j];
    }

    Out[48] ^= W1[32];
    Out[49] ^= W1[33];
    Out[64] ^= W2[32];
    Out[65] ^= W2[33];
    Out[80] ^= W3[32];
}

//len = 139: TC3_128
static inline void gfmul_139(poly8x16_t *Out,  const poly8x16_t *A256,  const poly8x16_t *B256){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[94], W1[95], W2[96], W3[96], W4[90], tmp[96];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (const poly8x16_t *)&A256[0];
    U1 = (const poly8x16_t *)&A256[47];
    U2 = (const poly8x16_t *)&A256[94];
    V0 = (const poly8x16_t *)&B256[0];
    V1 = (const poly8x16_t *)&B256[47];
    V2 = (const poly8x16_t *)&B256[94];

    for (int32_t i = 0 ; i < 45 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[45] = U0[45] ^ U1[45];
    W2[45] = V0[45] ^ V1[45];
    W3[46] = U0[46] ^ U1[46];
    W2[46] = V0[46] ^ V1[46];

    gfmul_47(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 46 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    W0[47] = U1[46];
    W4[47] = V1[46];

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
    for (int32_t i = 0 ; i < 96 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_48(W2, W0, W4);

    gfmul_45(W4, U2, V2);

    gfmul_47(W0, U0, V0);

    for (int32_t i = 0 ; i < 96 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 94 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 93 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[93] = W2[94];
    W2[94] = W2[95];

    for (int32_t i = 0 ; i < 90 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 90 ; i < 95 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[95] = W3[95];
    for (int32_t i = 0 ; i < 90 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 96);

    for (int32_t i = 0 ; i < 93 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[93] = W3[94];
    tmp[94] = W3[95];
    divide_by_x_plus_one_128(tmp, W3, 95);

    for (int32_t i = 0 ; i < 90 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 90 ; i < 94 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[94] = W2[94];

    for (int32_t i = 0 ; i < 94 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 43; i++) {
        int32_t j = i + 47;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 47] = W1[j] ^ W2[i];
        Out[j + 94] = W2[j] ^ W3[i];
        Out[i + 188] = W3[j] ^ W4[i];
        Out[j + 188] = W4[j];
    }
    for (int32_t i = 43; i < 47; i++) {
        int32_t j = i + 47;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 47] = W1[j] ^ W2[i];
        Out[j + 94] = W2[j] ^ W3[i];
        Out[i + 188] = W3[j] ^ W4[i];
    }
    Out[141] ^= W1[94];
    Out[188] ^= W2[94];
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
    uint64_t a11[278], a22[278];
    uint64_t o_karat[556];
    
    memcpy(a11,a1,277*8);
    memcpy(a22,a2,277*8);
    a11[277]=0;
    a22[277]=0;
    
    gfmul_139((poly8x16_t *)o_karat, (const poly8x16_t *)a11, (const poly8x16_t *)a22);
    reduce(o, o_karat);
}
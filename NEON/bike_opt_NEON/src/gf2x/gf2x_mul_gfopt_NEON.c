
#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include "gf2x_gfopt.h"


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

//len = 1: karat_NEON
static inline void gfmul_1(poly8x16_t* Out, const poly8x16_t* a, const poly8x16_t* b){
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
    D1 = (poly8x16_t)vmull64_a72((uint64_t)alpah, (uint64_t)blpbh);
    D0 = (poly8x16_t)vmull64_a72((uint64_t)al, (uint64_t)bl);
    D2 = (poly8x16_t)vmull64_a72((uint64_t)ah, (uint64_t)bh);

    D1=vaddq_p8(D1,D0);
    D1=vaddq_p8(D1,D2);

    Out[0]=vcombine_p8(vget_low_p8(D0),vadd_p8(vget_high_p8(D0),vget_low_p8(D1)));
    Out[1]=vcombine_p8(vadd_p8(vget_low_p8(D2),vget_high_p8(D1)),vget_high_p8(D2));
}


//len = 2: 2-Karatsuba
static inline void gfmul_2(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
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
static inline void gfmul_3(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
    const poly8x16_t *a0, *a1, *a2, *b0, *b1, *b2;
    static poly8x16_t aa01[1], bb01[1], aa02[1], bb02[1], aa12[1], bb12[1];
    static poly8x16_t D0[2], D1[2], D2[2], D3[2], D4[2], D5[2], middle;

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
static inline void gfmul_4(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
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

//len = 6: 2-Karatsuba
static inline void gfmul_6(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
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
static inline void gfmul_7(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
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
static inline void gfmul_8(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
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
    static poly8x16_t aa01[3], bb01[3], aa02[3], bb02[3], aa12[3], bb12[3];
    static poly8x16_t D0[6], D1[6], D2[6], D3[6], D4[6], D5[6], middle;

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

//len = 11: 3-Karatsuba
static inline void gfmul_11(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
    const  poly8x16_t *a0, *a1, *a2, *b0, *b1, *b2;
    static poly8x16_t aa01[4], bb01[4], aa02[4], bb02[4], aa12[4], bb12[4];
    static poly8x16_t D0[8], D1[8], D2[8], D3[8], D4[8], D5[8], middle;

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
static inline void gfmul_12(poly8x16_t *Out, const  poly8x16_t *A,  const  poly8x16_t *B){
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

//len = 13: 2-Karatsuba
static inline void gfmul_13(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
    static poly8x16_t D0[14], D1[14], D2[12], SAA[7], SBB[7];

    gfmul_7(D0, A, B);
    gfmul_6(D2, (A+7), (B+7));

    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 7;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    SAA[6]=A[6];
    SBB[6]=B[6];

    gfmul_7(D1, SAA, SBB);

    for(int32_t i = 0; i < 5; i++) {
        int32_t is = i + 7;
        int32_t is2 = is + 7;
        int32_t is3 = is2 + 7;
        poly8x16_t middle = D0[is] ^ D2[i];
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 5; i < 7; i++) {
        int32_t is = i + 7;
        int32_t is2 = is + 7;
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

//len = 21: TC3_64
static inline void gfmul_21(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[14], W1[16], W4[14];
    static poly8x16_t W2[16], W3[16], tmp[16];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[7];
    U2 = (const poly8x16_t *)&A128[14];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[7];
    V2 = (const poly8x16_t *)&B128[14];

    for (int32_t i = 0 ; i < 7 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_7(W1, W2, W3);

    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(poly8x8_t)V1_64[0]);
    U1_64 = ((const uint64_t *) U1) + 1;
    V1_64 = ((const uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 6; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[7] = (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[12],zero) ^ U2[6];
    W4[7] = (poly8x16_t)vcombine_p8((const poly8x8_t)V1_64[12],zero) ^ V2[6];

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

    gfmul_8(tmp, W3, W2);

    for (int32_t i = 0 ; i < 16; i++) {
        W3[i] = tmp[i];
    }

    gfmul_8(W2, W0, W4);
    gfmul_7(W4, U2, V2);
    gfmul_7(W0, U0, V0);

    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 14 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((const uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 13; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U2_64[i2])));
    }
    W2[13]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[26],zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[26])));
    W2[14]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[28])));
    W2[15]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[30],zero);

    U1_64 = ((const uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (const poly8x8_t)U1_64[0]);
    U1_64 = ((const uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 14; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2-4])));
    }
    tmp[14] = W2[14] ^ W3[14] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[24])));
    tmp[15] = W2[15] ^ W3[15] ^ (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[26], zero);
    divide_by_x_plus_one_64(W2, tmp, 32);

    U1_64 = ((const uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 13; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[13]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[26], zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[26])));
    tmp[14]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[28])));
    tmp[15]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[30], zero);
    divide_by_x_plus_one_64(W3, tmp, 31);

    for (int32_t i = 0 ; i < 14 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[14] = W2[14];
    W1[15] = W2[15];

    for (int32_t i = 0 ; i < 15 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 7; i++)
    {
        int32_t j = i + 7;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 7] = W1[j] ^ W2[i];
        Out[j + 14] = W2[j] ^ W3[i];
        Out[i + 28] = W3[j] ^ W4[i];
        Out[j + 28] = W4[j];
    }

    Out[21] ^= W1[14];
    Out[22] ^= W1[15];
    Out[28] ^= W2[14];
    Out[29] ^= W2[15];
    Out[35] ^= W3[14];
}

//len = 22: TC3_64
static inline void gfmul_22(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    poly8x16_t U0[8], U1[8], U2[7], V0[8], V1[8], V2[7];
    poly8x16_t W0[16], W1[16], W2[16], W3[17], W4[14];
    poly8x16_t tmp[17];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    const  uint64_t *A = (const  uint64_t *) A128;
    const  uint64_t *B = (const  uint64_t *) B128;

    for(int32_t i = 0; i < 7; i++) {
        int32_t i2 = i << 1;
        U0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2])));
        V0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2])));
        U1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 15])));
        V1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 15])));
        U2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 30])));
        V2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 30])));
    }
    U0[7]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[14], zero);
    V0[7]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[14], zero);
    U1[7]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[29], zero);
    V1[7]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[29], zero);

    for (int32_t i = 0 ; i < 7 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[7] = U0[7] ^ U1[7];
    W2[7] = V0[7] ^ V1[7];

    gfmul_8(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 7; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }

    for (int32_t i = 0 ; i < 8 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }

    for (int32_t i = 0 ; i < 8 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_8(tmp, W3, W2);

    for (int32_t i = 0 ; i < 16; i++) {
        W3[i] = tmp[i];
    }

    gfmul_8(W2, W0, W4);
    gfmul_7(W4, U2, V2);
    gfmul_8(W0, U0, V0);

    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 15 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 14; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[14]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[28], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[28])));
    W2[15]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[30], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 14; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[14] = W2[14] ^ W3[14] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[24])));
    tmp[15] = W2[15] ^ W3[15] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[26], zero);
    divide_by_x_plus_one_64(W2, tmp, 32);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 14; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[14]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[28], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[28])));
    tmp[15]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[30], zero);
    divide_by_x_plus_one_64(W3, tmp, 31);

    for (int32_t i = 0 ; i < 14 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[14] ^= W2[14];
    W1[15] = W2[15];

    for (int32_t i = 0 ; i < 15 ; i++) {
        W2[i] ^= W3[i];
    }

    for(int32_t i = 0; i < 14; i++) {
        Out[i] = W0[i];
        Out[i + 15] = W2[i];
        Out[i + 30] = W4[i];
    }
    Out[14] = W0[14];
    Out[29] = W2[14];
    Out[30] ^= W2[15];

    U1_64 = ((uint64_t *) &Out[7]) + 1;
    U2_64 = ((uint64_t *) &Out[22]) + 1;
    poly8x16_t aux;
    for(int32_t i = 0; i < 16; i++) {
        int32_t i2 = i << 1;
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];
        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];
        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));
    }
}

//len = 23: TC3_128
static inline void gfmul_23(poly8x16_t *Out,  const  poly8x16_t *A256,  const  poly8x16_t *B256){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[16], W1[17], W2[18], W3[18], W4[14], tmp[18];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (const poly8x16_t *)&A256[0];
    U1 = (const poly8x16_t *)&A256[8];
    U2 = (const poly8x16_t *)&A256[16];
    V0 = (const poly8x16_t *)&B256[0];
    V1 = (const poly8x16_t *)&B256[8];
    V2 = (const poly8x16_t *)&B256[16];

    for (int32_t i = 0 ; i < 7 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[7] = U0[7] ^ U1[7];
    W2[7] = V0[7] ^ V1[7];

    gfmul_8(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 8 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }

    for (int32_t i = 0 ; i < 8 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[8] = W0[8];
    W2[8] = W4[8];

    for (int32_t i = 0 ; i < 8 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_9(tmp, W3, W2);
    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_9(W2, W0, W4);

    gfmul_7(W4, U2, V2);

    gfmul_8(W0, U0, V0);

    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 16 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 15 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[15] = W2[16];
    W2[16] = W2[17];

    for (int32_t i = 0 ; i < 14 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 14 ; i < 17 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[17] = W3[17];
    for (int32_t i = 0 ; i < 14 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 18);

    for (int32_t i = 0 ; i < 15 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[15] = W3[16];
    tmp[16] = W3[17];
    divide_by_x_plus_one_128(tmp, W3, 17);

    for (int32_t i = 0 ; i < 14 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    for (int32_t i = 14 ; i < 16 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[16] = W2[16];

    for (int32_t i = 0 ; i < 16 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 6; i++) {
        int32_t j = i + 8;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 8] = W1[j] ^ W2[i];
        Out[j + 16] = W2[j] ^ W3[i];
        Out[i + 32] = W3[j] ^ W4[i];
        Out[j + 32] = W4[j];
    }
    for (int32_t i = 6; i < 8; i++) {
        int32_t j = i + 8;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 8] = W1[j] ^ W2[i];
        Out[j + 16] = W2[j] ^ W3[i];
        Out[i + 32] = W3[j] ^ W4[i];
    }
    Out[24] ^= W1[16];
    Out[32] ^= W2[16];
}

//len = 32: 2-Karatsuba
static inline void gfmul_32(poly8x16_t *Out,  const  poly8x16_t *A,  const  poly8x16_t *B){
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
static inline void gfmul_33(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[22], W1[24], W4[22];
    static poly8x16_t W2[24], W3[24], tmp[24];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[11];
    U2 = (const poly8x16_t *)&A128[22];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[11];
    V2 = (const poly8x16_t *)&B128[22];

    for (int32_t i = 0 ; i < 11 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_11(W1, W2, W3);

    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(const poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(const poly8x8_t)V1_64[0]);
    U1_64 = ((const uint64_t *) U1) + 1;
    V1_64 = ((const uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 10; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[11] = (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[20],zero) ^ U2[10];
    W4[11] = (poly8x16_t)vcombine_p8((const poly8x8_t)V1_64[20],zero) ^ V2[10];

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

    U1_64 = ((const uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U2_64[i2])));
    }
    W2[21]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[42],zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[42])));
    W2[22]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[44])));
    W2[23]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[46],zero);

    U1_64 = ((const uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (const poly8x8_t)U1_64[0]);
    U1_64 = ((const uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 22; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2-4])));
    }
    tmp[22] = W2[22] ^ W3[22] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[40])));
    tmp[23] = W2[23] ^ W3[23] ^ (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[42], zero);
    divide_by_x_plus_one_64(W2, tmp, 48);

    U1_64 = ((const uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[21]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[42], zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[42])));
    tmp[22]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[44])));
    tmp[23]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[46], zero);
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

//len = 35: TC3_128
static inline void gfmul_35(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[24], W1[25], W2[26], W3[26], W4[22], tmp[26];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[12];
    U2 = (const poly8x16_t *)&A128[24];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[12];
    V2 = (const poly8x16_t *)&B128[24];

    for (int32_t i = 0 ; i < 11 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[11] = U0[11] ^ U1[11];
    W2[11] = V0[11] ^ V1[11];

    gfmul_12(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 12 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }

    for (int32_t i = 0 ; i < 12 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[12] = W0[12];
    W2[12] = W4[12];

    for (int32_t i = 0 ; i < 12 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_13(tmp, W3, W2);
    for (int32_t i = 0 ; i < 26 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_13(W2, W0, W4);

    gfmul_11(W4, U2, V2);

    gfmul_12(W0, U0, V0);

    for (int32_t i = 0 ; i < 26 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 24 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 23 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[23] = W2[24];
    W2[24] = W2[25];

    for (int32_t i = 0 ; i < 22 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 22 ; i < 25 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[25] = W3[25];
    for (int32_t i = 0 ; i < 22 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 26);

    for (int32_t i = 0 ; i < 23 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[23] = W3[24];
    tmp[24] = W3[25];
    divide_by_x_plus_one_128(tmp, W3, 25);

    for (int32_t i = 0 ; i < 22 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    for (int32_t i = 22 ; i < 24 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[24] = W2[24];

    for (int32_t i = 0 ; i < 24 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 10; i++) {
        int32_t j = i + 12;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 12] = W1[j] ^ W2[i];
        Out[j + 24] = W2[j] ^ W3[i];
        Out[i + 48] = W3[j] ^ W4[i];
        Out[j + 48] = W4[j];
    }
    for (int32_t i = 10; i < 12; i++) {
        int32_t j = i + 12;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 12] = W1[j] ^ W2[i];
        Out[j + 24] = W2[j] ^ W3[i];
        Out[i + 48] = W3[j] ^ W4[i];
    }
    Out[36] ^= W1[24];
    Out[48] ^= W2[24];
}

//len = 36: TC3_64
static inline void gfmul_36(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[24], W1[26], W4[24];
    static poly8x16_t W2[26], W3[26], tmp[26];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[12];
    U2 = (const poly8x16_t *)&A128[24];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[12];
    V2 = (const poly8x16_t *)&B128[24];

    for (int32_t i = 0 ; i < 12 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_12(W1, W2, W3);

    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(const poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(const poly8x8_t)V1_64[0]);
    U1_64 = ((const uint64_t *) U1) + 1;
    V1_64 = ((const uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 11; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[12] = (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[22],zero) ^ U2[11];
    W4[12] = (poly8x16_t)vcombine_p8((const poly8x8_t)V1_64[22],zero) ^ V2[11];

    for (int32_t i = 0 ; i < 12 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[12] = W0[12];
    W2[12] = W4[12];

    for (int32_t i = 0 ; i < 12 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_13(tmp, W3, W2);

    for (int32_t i = 0 ; i < 26; i++) {
        W3[i] = tmp[i];
    }

    gfmul_13(W2, W0, W4);
    gfmul_12(W4, U2, V2);
    gfmul_12(W0, U0, V0);

    for (int32_t i = 0 ; i < 26 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 24 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((const uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 23; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U2_64[i2])));
    }
    W2[23]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[46],zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[46])));
    W2[24]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[48])));
    W2[25]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[50],zero);

    U1_64 = ((const uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (const poly8x8_t)U1_64[0]);
    U1_64 = ((const uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 24; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2-4])));
    }
    tmp[24] = W2[24] ^ W3[24] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[44])));
    tmp[25] = W2[25] ^ W3[25] ^ (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[46], zero);
    divide_by_x_plus_one_64(W2, tmp, 52);

    U1_64 = ((const uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 23; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[23]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[46], zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[46])));
    tmp[24]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[48])));
    tmp[25]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[50], zero);
    divide_by_x_plus_one_64(W3, tmp, 51);

    for (int32_t i = 0 ; i < 24 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[24] = W2[24];
    W1[25] = W2[25];

    for (int32_t i = 0 ; i < 25 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 12; i++)
    {
        int32_t j = i + 12;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 12] = W1[j] ^ W2[i];
        Out[j + 24] = W2[j] ^ W3[i];
        Out[i + 48] = W3[j] ^ W4[i];
        Out[j + 48] = W4[j];
    }

    Out[36] ^= W1[24];
    Out[37] ^= W1[25];
    Out[48] ^= W2[24];
    Out[49] ^= W2[25];
    Out[60] ^= W3[24];
}

//len = 37: TC3_64
static inline void gfmul_37(poly8x16_t *Out,   const poly8x16_t *A128,   const poly8x16_t *B128){
    poly8x16_t U0[13], U1[13], U2[12], V0[13], V1[13], V2[12];
    poly8x16_t W0[26], W1[26], W2[26], W3[27], W4[24];
    poly8x16_t tmp[27];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    const uint64_t *A = (const uint64_t *) A128;
    const uint64_t *B = (const uint64_t *) B128;

    for(int32_t i = 0; i < 12; i++) {
        int32_t i2 = i << 1;
        U0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2])));
        V0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2])));
        U1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 25])));
        V1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 25])));
        U2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 50])));
        V2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 50])));
    }
    U0[12]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[24], zero);
    V0[12]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[24], zero);
    U1[12]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[49], zero);
    V1[12]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[49], zero);

    for (int32_t i = 0 ; i < 12 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[12] = U0[12] ^ U1[12];
    W2[12] = V0[12] ^ V1[12];

    gfmul_13(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 12; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }

    for (int32_t i = 0 ; i < 13 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }

    for (int32_t i = 0 ; i < 13 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_13(tmp, W3, W2);

    for (int32_t i = 0 ; i < 26; i++) {
        W3[i] = tmp[i];
    }

    gfmul_13(W2, W0, W4);
    gfmul_12(W4, U2, V2);
    gfmul_13(W0, U0, V0);

    for (int32_t i = 0 ; i < 26 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 25 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 24; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[24]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[48], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[48])));
    W2[25]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[50], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 24; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[24] = W2[24] ^ W3[24] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[44])));
    tmp[25] = W2[25] ^ W3[25] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[46], zero);
    divide_by_x_plus_one_64(W2, tmp, 52);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 24; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[24]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[48], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[48])));
    tmp[25]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[50], zero);
    divide_by_x_plus_one_64(W3, tmp, 51);

    for (int32_t i = 0 ; i < 24 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[24] ^= W2[24];
    W1[25] = W2[25];

    for (int32_t i = 0 ; i < 25 ; i++) {
        W2[i] ^= W3[i];
    }

    for(int32_t i = 0; i < 24; i++) {
        Out[i] = W0[i];
        Out[i + 25] = W2[i];
        Out[i + 50] = W4[i];
    }
    Out[24] = W0[24];
    Out[49] = W2[24];
    Out[50] ^= W2[25];

    U1_64 = ((uint64_t *) &Out[12]) + 1;
    U2_64 = ((uint64_t *) &Out[37]) + 1;
    poly8x16_t aux;
    for(int32_t i = 0; i < 26; i++) {
        int32_t i2 = i << 1;
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];
        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];
        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));
    }
}

//len = 64: TC3_64
static inline void gfmul_64(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    poly8x16_t U0[22], U1[22], U2[21], V0[22], V1[22], V2[21];
    poly8x16_t W0[44], W1[44], W2[44], W3[45], W4[42];
    poly8x16_t tmp[45];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    const uint64_t *A = (const uint64_t *) A128;
    const uint64_t *B = (const uint64_t *) B128;

    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        U0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2])));
        V0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2])));
        U1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 43])));
        V1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 43])));
        U2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 86])));
        V2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 86])));
    }
    U0[21]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[42], zero);
    V0[21]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[42], zero);
    U1[21]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[85], zero);
    V1[21]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[85], zero);

    for (int32_t i = 0 ; i < 21 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[21] = U0[21] ^ U1[21];
    W2[21] = V0[21] ^ V1[21];

    gfmul_22(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }

    for (int32_t i = 0 ; i < 22 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }

    for (int32_t i = 0 ; i < 22 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_22(tmp, W3, W2);

    for (int32_t i = 0 ; i < 44; i++) {
        W3[i] = tmp[i];
    }

    gfmul_22(W2, W0, W4);
    gfmul_21(W4, U2, V2);
    gfmul_22(W0, U0, V0);

    for (int32_t i = 0 ; i < 44 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 43 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 42; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[42]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[84], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[84])));
    W2[43]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[86], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 42; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[42] = W2[42] ^ W3[42] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[80])));
    tmp[43] = W2[43] ^ W3[43] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[82], zero);
    divide_by_x_plus_one_64(W2, tmp, 88);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 42; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[42]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[84], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[84])));
    tmp[43]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[86], zero);
    divide_by_x_plus_one_64(W3, tmp, 87);

    for (int32_t i = 0 ; i < 42 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[42] ^= W2[42];
    W1[43] = W2[43];

    for (int32_t i = 0 ; i < 43 ; i++) {
        W2[i] ^= W3[i];
    }

    for(int32_t i = 0; i < 42; i++) {
        Out[i] = W0[i];
        Out[i + 43] = W2[i];
        Out[i + 86] = W4[i];
    }
    Out[42] = W0[42];
    Out[85] = W2[42];
    Out[86] ^= W2[43];

    U1_64 = ((uint64_t *) &Out[21]) + 1;
    U2_64 = ((uint64_t *) &Out[64]) + 1;
    poly8x16_t aux;
    for(int32_t i = 0; i < 44; i++) {
        int32_t i2 = i << 1;
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];
        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];
        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));
    }
}

//len = 65: TC3_64
static inline void gfmul_65(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[44], W1[45], W4[42];
    static poly8x16_t W2[46], W3[45], tmp[46];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[22];
    U2 = (const poly8x16_t *)&A128[44];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[22];
    V2 = (const poly8x16_t *)&B128[44];

    for (int32_t i = 0 ; i < 21 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    W3[21] = U0[21] ^ U1[21];
    W2[21] = V0[21] ^ V1[21];

    gfmul_22(W1, W2, W3);

    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (const poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (const poly8x8_t)V1_64[0]);

    U1_64 = ((const uint64_t *) U1) + 1;
    V1_64 = ((const uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 21; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }

    W0[22] = (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[42], zero);
    W4[22] = (poly8x16_t)vcombine_p8((const poly8x8_t)V1_64[42], zero);

    for (int32_t i = 0 ; i < 22 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[22] = W0[22];
    W2[22] = W4[22];

    for (int32_t i = 0 ; i < 22 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_23(tmp, W3, W2);

    for (int32_t i = 0 ; i < 45; i++) {
        W3[i] = tmp[i];
    }

    gfmul_23(W2, W0, W4);
    gfmul_21(W4, U2, V2);
    gfmul_22(W0, U0, V0);

    for (int32_t i = 0 ; i < 45 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 44 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((const uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;

    for(int32_t i = 0; i < 43; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[43]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[86]))) ^ (poly8x16_t)vcombine_p8((poly8x8_t)U2_64[86], zero);
    W2[44]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[88], zero);

    U1_64 = ((const uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (const poly8x8_t)U1_64[0]);

    U1_64 = ((const uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 42; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2-4])));
    }
    tmp[42] = W2[42] ^ W3[42] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[80])));
    tmp[43] = W2[43] ^ W3[43] ^ (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[82], zero);
    tmp[44] = W2[44] ^ W3[44];
    divide_by_x_plus_one_64(W2, tmp, 90);

    U1_64 = ((const uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 43; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }

    tmp[43]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[86]))) ^ (poly8x16_t)vcombine_p8((poly8x8_t)U2_64[86], zero);
    tmp[44]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[88], zero);
    divide_by_x_plus_one_64(W3, tmp, 89);

    for (int32_t i = 0 ; i < 42 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[42] ^= W2[42];
    W1[43] ^= W2[43];
    W1[44] = W2[44];

    for (int32_t i = 0 ; i < 44 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 20; i++)
    {
        int32_t j = i + 22;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 22] = W1[j] ^ W2[i];
        Out[j + 44] = W2[j] ^ W3[i];
        Out[i + 88] = W3[j] ^ W4[i];
        Out[j + 88] = W4[j];
    }

    for (int32_t i = 20; i < 22; i++)
    {
        int32_t j = i + 22;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 22] = W1[j] ^ W2[i];
        Out[j + 44] = W2[j] ^ W3[i];
        Out[i + 88] = W3[j] ^ W4[i];
    }
    Out[66] ^= W1[44];
    Out[88] ^= W2[44];
}


//len = 107: TC3_128
static inline void gfmul_107(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[72], W1[73], W2[74], W3[74], W4[70], tmp[74];
    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[36];
    U2 = (const poly8x16_t *)&A128[72];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[36];
    V2 = (const poly8x16_t *)&B128[72];

    for (int32_t i = 0 ; i < 35 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[35] = U0[35] ^ U1[35];
    W2[35] = V0[35] ^ V1[35];

    gfmul_36(W1, W2, W3);

    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 36 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }

    for (int32_t i = 0 ; i < 36 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[36] = W0[36];
    W2[36] = W4[36];

    for (int32_t i = 0 ; i < 36 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_37(tmp, W3, W2);
    for (int32_t i = 0 ; i < 74 ; i++) {
        W3[i] = tmp[i];
    }

    gfmul_37(W2, W0, W4);

    gfmul_35(W4, U2, V2);

    gfmul_36(W0, U0, V0);

    for (int32_t i = 0 ; i < 74 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 72 ; i++) {
        W1[i] ^= W0[i];
    }

    for (int32_t i = 0 ; i < 71 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[71] = W2[72];
    W2[72] = W2[73];

    for (int32_t i = 0 ; i < 70 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 70 ; i < 73 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[73] = W3[73];
    for (int32_t i = 0 ; i < 70 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_128(tmp, W2, 74);

    for (int32_t i = 0 ; i < 71 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[71] = W3[72];
    tmp[72] = W3[73];
    divide_by_x_plus_one_128(tmp, W3, 73);

    for (int32_t i = 0 ; i < 70 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }

    for (int32_t i = 70 ; i < 72 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[72] = W2[72];

    for (int32_t i = 0 ; i < 72 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 34; i++) {
        int32_t j = i + 36;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 36] = W1[j] ^ W2[i];
        Out[j + 72] = W2[j] ^ W3[i];
        Out[i + 144] = W3[j] ^ W4[i];
        Out[j + 144] = W4[j];
    }
    for (int32_t i = 34; i < 36; i++) {
        int32_t j = i + 36;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 36] = W1[j] ^ W2[i];
        Out[j + 72] = W2[j] ^ W3[i];
        Out[i + 144] = W3[j] ^ W4[i];
    }
    Out[108] ^= W1[72];
    Out[144] ^= W2[72];
}

//len = 108: TC3_64
static inline void gfmul_108(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[72], W1[74], W4[72];
    static poly8x16_t W2[74], W3[74], tmp[74];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[36];
    U2 = (const poly8x16_t *)&A128[72];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[36];
    V2 = (const poly8x16_t *)&B128[72];

    for (int32_t i = 0 ; i < 36 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_36(W1, W2, W3);

    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(const poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(const poly8x8_t)V1_64[0]);
    U1_64 = ((const uint64_t *) U1) + 1;
    V1_64 = ((const uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 35; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[36] = (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[70],zero) ^ U2[35];
    W4[36] = (poly8x16_t)vcombine_p8((const poly8x8_t)V1_64[70],zero) ^ V2[35];

    for (int32_t i = 0 ; i < 36 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[36] = W0[36];
    W2[36] = W4[36];

    for (int32_t i = 0 ; i < 36 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_37(tmp, W3, W2);

    for (int32_t i = 0 ; i < 74; i++) {
        W3[i] = tmp[i];
    }

    gfmul_37(W2, W0, W4);
    gfmul_36(W4, U2, V2);
    gfmul_36(W0, U0, V0);

    for (int32_t i = 0 ; i < 74 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 72 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((const uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 71; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U2_64[i2])));
    }
    W2[71]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[142],zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[142])));
    W2[72]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[144])));
    W2[73]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[146],zero);

    U1_64 = ((const uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (const poly8x8_t)U1_64[0]);
    U1_64 = (const uint64_t *)((uint64_t *)W4 + 1);
    for(int32_t i = 2; i < 72; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2-4])));
    }
    tmp[72] = W2[72] ^ W3[72] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[140])));
    tmp[73] = W2[73] ^ W3[73] ^ (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[142], zero);
    divide_by_x_plus_one_64(W2, tmp, 148);

    U1_64 = (const uint64_t *)((uint64_t *) W3 + 1);
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 71; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[71]=(poly8x16_t)vcombine_p8((const poly8x8_t)U2_64[142], zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[142])));
    tmp[72]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[144])));
    tmp[73]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[146], zero);
    divide_by_x_plus_one_64(W3, tmp, 147);

    for (int32_t i = 0 ; i < 72 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[72] = W2[72];
    W1[73] = W2[73];

    for (int32_t i = 0 ; i < 73 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 36; i++)
    {
        int32_t j = i + 36;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 36] = W1[j] ^ W2[i];
        Out[j + 72] = W2[j] ^ W3[i];
        Out[i + 144] = W3[j] ^ W4[i];
        Out[j + 144] = W4[j];
    }

    Out[108] ^= W1[72];
    Out[109] ^= W1[73];
    Out[144] ^= W2[72];
    Out[145] ^= W2[73];
    Out[180] ^= W3[72];
}

//=============================================================

//len = 97: TC3_64
void gfmul_97(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    poly8x16_t U0[33], U1[33], U2[32], V0[33], V1[33], V2[32];
    poly8x16_t W0[66], W1[66], W2[66], W3[67], W4[64];
    poly8x16_t tmp[67];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    const uint64_t *A = (const uint64_t *) A128;
    const uint64_t *B = (const uint64_t *) B128;

    for(int32_t i = 0; i < 32; i++) {
        int32_t i2 = i << 1;
        U0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2])));
        V0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2])));
        U1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 65])));
        V1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 65])));
        U2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 130])));
        V2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 130])));
    }
    U0[32]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[64], zero);
    V0[32]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[64], zero);
    U1[32]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[129], zero);
    V1[32]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[129], zero);

    for (int32_t i = 0 ; i < 32 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[32] = U0[32] ^ U1[32];
    W2[32] = V0[32] ^ V1[32];

    gfmul_33(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 32; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
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

    gfmul_33(tmp, W3, W2);

    for (int32_t i = 0 ; i < 66; i++) {
        W3[i] = tmp[i];
    }

    gfmul_33(W2, W0, W4);
    gfmul_32(W4, U2, V2);
    gfmul_33(W0, U0, V0);

    for (int32_t i = 0 ; i < 66 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 65 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 64; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[64]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[128], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[128])));
    W2[65]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[130], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 64; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[64] = W2[64] ^ W3[64] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[124])));
    tmp[65] = W2[65] ^ W3[65] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[126], zero);
    divide_by_x_plus_one_64(W2, tmp, 132);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 64; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[64]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[128], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[128])));
    tmp[65]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[130], zero);
    divide_by_x_plus_one_64(W3, tmp, 131);

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

    U1_64 = ((uint64_t *) &Out[32]) + 1;
    U2_64 = ((uint64_t *) &Out[97]) + 1;
    poly8x16_t aux;
    for(int32_t i = 0; i < 66; i++) {
        int32_t i2 = i << 1;
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];
        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];
        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));
    }
}

//len = 193: TC3_64
void gfmul_193(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128){
    poly8x16_t U0[65], U1[65], U2[64], V0[65], V1[65], V2[64];
    poly8x16_t W0[130], W1[130], W2[130], W3[131], W4[128];
    poly8x16_t tmp[131];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    const uint64_t *A = (const uint64_t *) A128;
    const uint64_t *B = (const uint64_t *) B128;

    for(int32_t i = 0; i < 64; i++) {
        int32_t i2 = i << 1;
        U0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2])));
        V0[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2])));
        U1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 129])));
        V1[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 129])));
        U2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& A[i2 + 258])));
        V2[i]= vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& B[i2 + 258])));
    }
    U0[64]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[128], zero);
    V0[64]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[128], zero);
    U1[64]= (poly8x16_t)vcombine_p8((const poly8x8_t)A[257], zero);
    V1[64]= (poly8x16_t)vcombine_p8((const poly8x8_t)B[257], zero);

    for (int32_t i = 0 ; i < 64 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[64] = U0[64] ^ U1[64];
    W2[64] = V0[64] ^ V1[64];

    gfmul_65(W1, W2, W3);

    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);
    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);

    U1_64 = ((uint64_t *) U1) + 1;
    V1_64 = ((uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 64; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }

    for (int32_t i = 0 ; i < 65 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }

    for (int32_t i = 0 ; i < 65 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_65(tmp, W3, W2);

    for (int32_t i = 0 ; i < 130; i++) {
        W3[i] = tmp[i];
    }

    gfmul_65(W2, W0, W4);
    gfmul_64(W4, U2, V2);
    gfmul_65(W0, U0, V0);

    for (int32_t i = 0 ; i < 130 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 129 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = ((uint64_t *) W2) + 1;
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 128; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[128]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[256], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[256])));
    W2[129]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[258], zero);

    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);

    U1_64 = ((uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 128; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));
    }
    tmp[128] = W2[128] ^ W3[128] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[252])));
    tmp[129] = W2[129] ^ W3[129] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[254], zero);
    divide_by_x_plus_one_64(W2, tmp, 260);

    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 128; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[128]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[256], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[256])));
    tmp[129]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[258], zero);
    divide_by_x_plus_one_64(W3, tmp, 259);

    for (int32_t i = 0 ; i < 128 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[128] ^= W2[128];
    W1[129] = W2[129];

    for (int32_t i = 0 ; i < 129 ; i++) {
        W2[i] ^= W3[i];
    }

    for(int32_t i = 0; i < 128; i++) {
        Out[i] = W0[i];
        Out[i + 129] = W2[i];
        Out[i + 258] = W4[i];
    }
    Out[128] = W0[128];
    Out[257] = W2[128];
    Out[258] ^= W2[129];

    U1_64 = ((uint64_t *) &Out[64]) + 1;
    U2_64 = ((uint64_t *) &Out[193]) + 1;
    poly8x16_t aux;
    for(int32_t i = 0; i < 130; i++) {
        int32_t i2 = i << 1;
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];
        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));
        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];
        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));
    }
}

//len = 321: TC3_64
void gfmul_321(poly8x16_t *Out,   const poly8x16_t *A128,   const poly8x16_t *B128){
    const poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;
    static poly8x16_t W0[214], W1[216], W4[214];
    static poly8x16_t W2[216], W3[216], tmp[216];
    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

    U0 = (const poly8x16_t *)&A128[0];
    U1 = (const poly8x16_t *)&A128[107];
    U2 = (const poly8x16_t *)&A128[214];
    V0 = (const poly8x16_t *)&B128[0];
    V1 = (const poly8x16_t *)&B128[107];
    V2 = (const poly8x16_t *)&B128[214];

    for (int32_t i = 0 ; i < 107 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }

    gfmul_107(W1, W2, W3);

    const uint64_t *U1_64 = ((const uint64_t *) U1);
    const uint64_t *V1_64 = ((const uint64_t *) V1);
    W0[0] = vcombine_p8(zero,(const poly8x8_t)U1_64[0]);
    W4[0] = vcombine_p8(zero,(const poly8x8_t)V1_64[0]);
    U1_64 = ((const uint64_t *) U1) + 1;
    V1_64 = ((const uint64_t *) V1) + 1;
    for(int32_t i = 0; i < 106; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W0[i1] ^= U2[i];
        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& V1_64[i2])));
        W4[i1] ^= V2[i];
    }
    W0[107] = (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[212],zero) ^ U2[106];
    W4[107] = (poly8x16_t)vcombine_p8((const poly8x8_t)V1_64[212],zero) ^ V2[106];

    for (int32_t i = 0 ; i < 107 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[107] = W0[107];
    W2[107] = W4[107];

    for (int32_t i = 0 ; i < 107 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }

    gfmul_108(tmp, W3, W2);

    for (int32_t i = 0 ; i < 216; i++) {
        W3[i] = tmp[i];
    }

    gfmul_108(W2, W0, W4);
    gfmul_107(W4, U2, V2);
    gfmul_107(W0, U0, V0);

    for (int32_t i = 0 ; i < 216 ; i++) {
        W3[i] ^= W2[i];
    }

    for (int32_t i = 0 ; i < 214 ; i++) {
        W1[i] ^= W0[i];
    }

    U1_64 = (const uint64_t *) ((uint64_t *) W2 + 1);
    uint64_t * U2_64 = ((uint64_t *) W0) + 1;
    for(int32_t i = 0; i < 213; i++) {
        int32_t i2 = i << 1;
        W2[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2])));
        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    W2[213]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[426],zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[426])));
    W2[214]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[428])));
    W2[215]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[430],zero);

    U1_64 = ((const uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (const poly8x8_t)U1_64[0]);
    U1_64 = ((const uint64_t *) W4) + 1;
    for(int32_t i = 2; i < 214; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2-4])));
    }
    tmp[214] = W2[214] ^ W3[214] ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[424])));
    tmp[215] = W2[215] ^ W3[215] ^ (poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[426], zero);
    divide_by_x_plus_one_64(W2, tmp, 432);

    U1_64 = ((const uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for(int32_t i = 0; i < 213; i++) {
        int32_t i2 = i << 1;
        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));
    }
    tmp[213]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[426], zero) ^ vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[426])));
    tmp[214]=vreinterpretq_p8_p128(vldrq_p128((const poly128_t *)(& U1_64[428])));
    tmp[215]=(poly8x16_t)vcombine_p8((const poly8x8_t)U1_64[430], zero);
    divide_by_x_plus_one_64(W3, tmp, 431);

    for (int32_t i = 0 ; i < 214 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[214] = W2[214];
    W1[215] = W2[215];

    for (int32_t i = 0 ; i < 215 ; i++) {
        W2[i] ^= W3[i];
    }

    for (int32_t i = 0; i < 107; i++)
    {
        int32_t j = i + 107;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 107] = W1[j] ^ W2[i];
        Out[j + 214] = W2[j] ^ W3[i];
        Out[i + 428] = W3[j] ^ W4[i];
        Out[j + 428] = W4[j];
    }

    Out[321] ^= W1[214];
    Out[322] ^= W1[215];
    Out[428] ^= W2[214];
    Out[429] ^= W2[215];
    Out[535] ^= W3[214];
}

//=============================================================

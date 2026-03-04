#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gfopt.h"
#include "best_alg/best_alg_result_NEON.h"

void fprint_divide_by_x_plus_one_128(FILE * fp){
    fprintf(fp,"static inline void divide_by_x_plus_one_128(poly8x16_t *in, poly8x16_t *out, int32_t size) {\n"); 
    fprintf(fp,"    out[0] = in[0];\n");
    fprintf(fp,"    for(int32_t i = 1 ; i < size ; i++) {\n");
    fprintf(fp,"        out[i] = out[i - 1] ^ in[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n\n");
}

void fprint_divide_by_x_plus_one_64(FILE * fp){
    fprintf(fp,"static inline void divide_by_x_plus_one_64(poly8x16_t *out,poly8x16_t *in,int32_t size){\n"); 
    fprintf(fp,"    uint64_t *A = (uint64_t *) in;\n");
    fprintf(fp,"    uint64_t *B = (uint64_t *) out;\n\n");
        
    fprintf(fp,"    B[0] = A[0];\n");
    fprintf(fp,"    for(int32_t i = 1; i < size; i++) {\n");
    fprintf(fp,"        B[i] = B[i - 1] ^ A[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n\n");
}

void fprint_karat_128_NEON(FILE * fp, char * func_name){
    fprintf(fp,"static inline void %s(poly8x16_t* Out, poly8x16_t* a, poly8x16_t* b){\n", func_name);
    fprintf(fp,"	poly8x8_t al, ah, bl, bh;\n");
    fprintf(fp,"	poly8x8_t alpah, blpbh;\n");
    fprintf(fp,"	poly8x16_t D0={0}, D1={0}, D2={0};\n");

    fprintf(fp,"	ah = vget_high_p8(*a);\n");
	  fprintf(fp,"	al = vget_low_p8(*a);\n");
    fprintf(fp,"	bh = vget_high_p8(*b);\n");
   	fprintf(fp,"	bl = vget_low_p8(*b);\n");

    fprintf(fp,"	alpah=vadd_p8(al,ah);\n");
    fprintf(fp,"	blpbh=vadd_p8(bl,bh);\n");
//    fprintf(fp,"    schmul64_NEON(&D1, &alpah, &blpbh);\n");
//    fprintf(fp,"    schmul64_NEON(&D0, &al, &bl);\n");
//    fprintf(fp,"    schmul64_NEON(&D2, &ah, &bh);\n\n");
    fprintf(fp,"    D1 = (poly8x16_t)vmull64_a72(*(uint64_t *)&alpah, *(uint64_t *)&blpbh);\n");
    fprintf(fp,"    D0 = (poly8x16_t)vmull64_a72(*(uint64_t *)&al, *(uint64_t *)&bl);\n");
    fprintf(fp,"    D2 = (poly8x16_t)vmull64_a72(*(uint64_t *)&ah, *(uint64_t *)&bh);\n\n");
    
    fprintf(fp,"    D1=vaddq_p8(D1,D0);\n");
    fprintf(fp,"    D1=vaddq_p8(D1,D2);\n\n");
    
    fprintf(fp,"    Out[0]=vcombine_p8(vget_low_p8(D0),vadd_p8(vget_high_p8(D0),vget_low_p8(D1)));\n");
    fprintf(fp,"    Out[1]=vcombine_p8(vadd_p8(vget_low_p8(D2),vget_high_p8(D1)),vget_high_p8(D2));\n");
    fprintf(fp,"}\n\n");
}

void fprint_SB_128_NEON(FILE * fp, char * func_name){
    fprintf(fp,"static inline void %s(poly8x16_t* Out, poly8x16_t* a, poly8x16_t* b){;\n", func_name);
    fprintf(fp,"	poly8x8_t al, ah, bl, bh;\n");
    fprintf(fp,"	poly8x16_t D0={0}, D1={0}, D2={0};\n\n");

    fprintf(fp,"	ah = vget_high_p8(*a);\n");
	  fprintf(fp,"	al = vget_low_p8(*a);\n");
    fprintf(fp,"	bh = vget_high_p8(*b);\n");
	  fprintf(fp,"	bl = vget_low_p8(*b);\n\n");

//    fprintf(fp,"	schmul64_NEON(&D0, &al, &bh);\n");
//    fprintf(fp,"	schmul64_NEON(&D1, &ah, &bl);\n");
    fprintf(fp,"  D0 = (poly8x16_t)vmull64_a72(*(uint64_t *)&al, *(uint64_t *)&bh);\n");
    fprintf(fp,"  D1 = (poly8x16_t)vmull64_a72(*(uint64_t *)&ah, *(uint64_t *)&bl);\n\n");
    
    fprintf(fp,"	D1=vaddq_p8(D0,D1);\n\n");

//    fprintf(fp,"	schmul64_NEON(&D0, &al, &bl);\n");
//    fprintf(fp,"	schmul64_NEON(&D2, &ah, &bh);\n\n");
    fprintf(fp,"	D0 = (poly8x16_t)vmull64_a72(*(uint64_t *)&al, *(uint64_t *)&bl);\n");
    fprintf(fp,"	D2 = (poly8x16_t)vmull64_a72(*(uint64_t *)&ah, *(uint64_t *)&bh);\n\n");
    

    fprintf(fp,"	Out[0]=vcombine_p8(vget_low_p8(D0),vadd_p8(vget_high_p8(D0),vget_low_p8(D1)));\n");
    fprintf(fp,"	Out[1]=vcombine_p8(vadd_p8(vget_low_p8(D2),vget_high_p8(D1)),vget_high_p8(D2));\n");
    fprintf(fp,"}\n\n");
}

//karat, Toom-Cook
void fprint_karat2_2k(FILE * fp, char * func_name, uint64_t len){
        uint64_t k = len / 2;
        fprintf(fp,"static inline void %s(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){\n",func_name);
        fprintf(fp,"    static poly8x16_t D0[%ld], D1[%ld], D2[%ld], SAA[%ld], SBB[%ld];\n",len,len,len,k,k);
        fprintf(fp,"\n");
        fprintf(fp,"    gfmul_%ld(D0, A, B);\n",k);
        fprintf(fp,"    gfmul_%ld(D2, (A+%ld), (B+%ld));\n",k,k,k);
        fprintf(fp,"\n");
        fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
        fprintf(fp,"        int32_t is = i + %ld;\n",k);
        fprintf(fp,"        SAA[i] = A[i] ^ A[is];\n");
        fprintf(fp,"        SBB[i] = B[i] ^ B[is];\n");
        fprintf(fp,"    }\n");
        fprintf(fp,"\n");
        fprintf(fp,"    gfmul_%ld(D1, SAA, SBB);\n",k);
        fprintf(fp,"\n");
        fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
        fprintf(fp,"        int32_t is = i + %ld;\n",k);
        fprintf(fp,"        int32_t is2 = is + %ld;\n",k);
        fprintf(fp,"        int32_t is3 = is2 + %ld;\n",k);
        fprintf(fp,"        poly8x16_t middle = D0[is] ^ D2[i];\n");
        fprintf(fp,"        Out[i]   = D0[i];\n");
        fprintf(fp,"        Out[is]  = middle ^ D0[i] ^ D1[i];\n");
        fprintf(fp,"        Out[is2] = middle ^ D1[is] ^ D2[is];\n");
        fprintf(fp,"        Out[is3] = D2[is];\n");
        fprintf(fp,"    }\n");
        fprintf(fp,"}\n");
        fprintf(fp,"\n");
}
void fprint_karat2_2kp1(FILE * fp, char * func_name, uint64_t len){
        uint64_t k = len / 2;
        fprintf(fp,"static inline void %s(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){\n",func_name);
        fprintf(fp,"    static poly8x16_t D0[%ld], D1[%ld], D2[%ld], SAA[%ld], SBB[%ld];\n",len+1,len+1,len-1,k+1,k+1);
        fprintf(fp,"\n");

        fprintf(fp,"    gfmul_%ld(D0, A, B);\n",k+1);
        fprintf(fp,"    gfmul_%ld(D2, (A+%ld), (B+%ld));\n",k,k+1,k+1);
        fprintf(fp,"\n");
        
        fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
        fprintf(fp,"        int32_t is = i + %ld;\n",k+1);
        fprintf(fp,"        SAA[i] = A[i] ^ A[is];\n");
        fprintf(fp,"        SBB[i] = B[i] ^ B[is];\n");
        fprintf(fp,"    }\n");
        fprintf(fp,"    SAA[%ld]=A[%ld];\n",k,k);
        fprintf(fp,"    SBB[%ld]=B[%ld];\n",k,k);
        fprintf(fp,"\n");

        fprintf(fp,"    gfmul_%ld(D1, SAA, SBB);\n",k+1);
        fprintf(fp,"\n");

        fprintf(fp,"    for(int32_t i = 0; i < %ld; i++) {\n",k-1);
        fprintf(fp,"        int32_t is = i + %ld;\n",k+1);
        fprintf(fp,"        int32_t is2 = is + %ld;\n",k+1);
        fprintf(fp,"        int32_t is3 = is2 + %ld;\n",k+1);
        fprintf(fp,"        poly8x16_t middle = D0[is] ^ D2[i];\n");
        fprintf(fp,"        Out[i]   = D0[i];\n");
        fprintf(fp,"        Out[is]  = middle ^ D0[i] ^ D1[i];\n");
        fprintf(fp,"        Out[is2] = middle ^ D1[is] ^ D2[is];\n");
        fprintf(fp,"        Out[is3] = D2[is];\n");
        fprintf(fp,"    }\n");
        fprintf(fp,"    for(int32_t i = %ld; i < %ld; i++) {\n",k-1,k+1);
        fprintf(fp,"        int32_t is = i + %ld;\n",k+1);
        fprintf(fp,"        int32_t is2 = is + %ld;\n",k+1);
        fprintf(fp,"        poly8x16_t middle = D0[is] ^ D2[i];\n");
        fprintf(fp,"        Out[i]   = D0[i];\n");
        fprintf(fp,"        Out[is]  = middle ^ D0[i] ^ D1[i];\n");
        fprintf(fp,"        Out[is2] = middle ^ D1[is];\n");
        fprintf(fp,"    }\n");
        fprintf(fp,"}\n");
        fprintf(fp,"\n");
}

void fprint_karat3_3k(FILE * fp, char * func_name, uint64_t len){
        uint64_t k=len/3;
        fprintf(fp,"static inline void %s(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){\n",func_name);
        fprintf(fp,"    static poly8x16_t *a0, *a1, *a2, *b0, *b1, *b2, middle;\n");
        fprintf(fp,"    static poly8x16_t aa01[%ld], bb01[%ld], aa02[%ld], bb02[%ld], aa12[%ld], bb12[%ld];\n",k,k,k,k,k,k);
        fprintf(fp,"    static poly8x16_t D0[%ld], D1[%ld], D2[%ld], D3[%ld], D4[%ld], D5[%ld];\n",2*k,2*k,2*k,2*k,2*k,2*k);
        fprintf(fp,"\n");

        fprintf(fp,"    a0 = A;\n");
        fprintf(fp,"    a1 = A + %ld;\n",k);
        fprintf(fp,"    a2 = A + %ld;\n",2*k);
        fprintf(fp,"    b0 = B;\n");
        fprintf(fp,"    b1 = B + %ld;\n",k);
        fprintf(fp,"    b2 = B + %ld;\n",2*k);
        fprintf(fp,"\n");
        fprintf(fp,"    for (int16_t i = 0; i < %ld; i++)\n",k);
        fprintf(fp,"    {\n");
        fprintf(fp,"        aa01[i] = a0[i] ^ a1[i];\n");
        fprintf(fp,"        bb01[i] = b0[i] ^ b1[i];\n");
        fprintf(fp,"        aa12[i] = a2[i] ^ a1[i];\n");
        fprintf(fp,"        bb12[i] = b2[i] ^ b1[i];\n");
        fprintf(fp,"        aa02[i] = a0[i] ^ a2[i];\n");
        fprintf(fp,"        bb02[i] = b0[i] ^ b2[i];\n");
        fprintf(fp,"    }\n");
        fprintf(fp,"\n");
        fprintf(fp,"    gfmul_%ld(D3, aa01, bb01);\n",k);
        fprintf(fp,"    gfmul_%ld(D4, aa02, bb02);\n",k);
        fprintf(fp,"    gfmul_%ld(D5, aa12, bb12);\n",k);
        fprintf(fp,"    gfmul_%ld(D0, a0, b0);\n",k);
        fprintf(fp,"    gfmul_%ld(D1, a1, b1);\n",k);
        fprintf(fp,"    gfmul_%ld(D2, a2, b2);\n",k);
        fprintf(fp,"\n");
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
        uint64_t k=len/3;
        fprintf(fp,"static inline void %s(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){\n",func_name);
        fprintf(fp,"    static poly8x16_t *a0, *a1, *a2, *b0, *b1, *b2, middle;\n");
        fprintf(fp,"    static poly8x16_t aa01[%ld], bb01[%ld], aa02[%ld], bb02[%ld], aa12[%ld], bb12[%ld];\n",k,k,k+1,k+1,k+1,k+1);
        fprintf(fp,"    static poly8x16_t D0[%ld], D1[%ld], D2[%ld], D3[%ld], D4[%ld], D5[%ld];\n",2*k,2*k,2*k+2,2*k,2*k+2,2*k+2);
        fprintf(fp,"\n");
        
        fprintf(fp,"    a0 = A;\n");
        fprintf(fp,"    a1 = A + %ld;\n",k);
        fprintf(fp,"    a2 = A + %ld;\n",2*k);
        fprintf(fp,"    b0 = B;\n");
        fprintf(fp,"    b1 = B + %ld;\n",k);
        fprintf(fp,"    b2 = B + %ld;\n",2*k);
        fprintf(fp,"\n");

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
        fprintf(fp,"\n");
        
        fprintf(fp,"    gfmul_%ld(D3, aa01, bb01);\n",k);
        fprintf(fp,"    gfmul_%ld(D4, aa02, bb02);\n",k+1);
        fprintf(fp,"    gfmul_%ld(D5, aa12, bb12);\n",k+1);
        fprintf(fp,"    gfmul_%ld(D0, a0, b0);\n",k);
        fprintf(fp,"    gfmul_%ld(D1, a1, b1);\n",k);
        fprintf(fp,"    gfmul_%ld(D2, a2, b2);\n",k+1);
        fprintf(fp,"\n");
        
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
    uint64_t k=len/3;
    fprintf(fp,"static inline void %s(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){\n",func_name);
    fprintf(fp,"    static poly8x16_t *a0, *a1, *a2, *b0, *b1, *b2, middle;\n");
    fprintf(fp,"    static poly8x16_t aa01[%ld], bb01[%ld], aa02[%ld], bb02[%ld], aa12[%ld], bb12[%ld];\n",k+1,k+1,k+1,k+1,k+1,k+1);
    fprintf(fp,"    static poly8x16_t D0[%ld], D1[%ld], D2[%ld], D3[%ld], D4[%ld], D5[%ld];\n",2*(k+1),2*(k+1),2*(k+1),2*(k+1),2*(k+1),2*(k+1));
    fprintf(fp,"\n");

    fprintf(fp,"    a0 = A;\n");
    fprintf(fp,"    a1 = A + %ld;\n",k+1);
    fprintf(fp,"    a2 = A + %ld;\n",2*(k+1));
    fprintf(fp,"    b0 = B;\n");
    fprintf(fp,"    b1 = B + %ld;\n",k+1);
    fprintf(fp,"    b2 = B + %ld;\n",2*(k+1));
    fprintf(fp,"\n");

    fprintf(fp,"    for (int16_t i = 0; i < %ld; i++)\n",k);
    fprintf(fp,"    {\n");
    fprintf(fp,"        aa01[i] = a0[i] ^ a1[i];\n");
    fprintf(fp,"        bb01[i] = b0[i] ^ b1[i];\n");
    fprintf(fp,"        aa12[i] = a2[i] ^ a1[i];\n");
    fprintf(fp,"        bb12[i] = b2[i] ^ b1[i];\n");
    fprintf(fp,"        aa02[i] = a0[i] ^ a2[i];\n");
    fprintf(fp,"        bb02[i] = b0[i] ^ b2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");

    fprintf(fp,"    aa01[%ld] = a0[%ld] ^ a1[%ld];\n",k,k,k);
    fprintf(fp,"    bb01[%ld] = b0[%ld] ^ b1[%ld];\n",k,k,k);
    fprintf(fp,"    aa12[%ld] = a1[%ld];\n",k,k);
    fprintf(fp,"    bb12[%ld] = b1[%ld];\n",k,k);
    fprintf(fp,"    aa02[%ld] = a0[%ld];\n",k,k);
    fprintf(fp,"    bb02[%ld] = b0[%ld];\n",k,k);
    fprintf(fp,"\n");

    fprintf(fp,"    gfmul_%ld(D3, aa01, bb01);\n",k+1);
    fprintf(fp,"    gfmul_%ld(D4, aa02, bb02);\n",k+1);
    fprintf(fp,"    gfmul_%ld(D5, aa12, bb12);\n",k+1);
    fprintf(fp,"    gfmul_%ld(D0, a0, b0);\n",k+1);
    fprintf(fp,"    gfmul_%ld(D1, a1, b1);\n",k+1);
    fprintf(fp,"    gfmul_%ld(D2, a2, b2);\n",k);
    fprintf(fp,"\n");

    fprintf(fp,"    for (int16_t i = 0; i < %ld; i++)\n",k-1);
    fprintf(fp,"    {\n");
    fprintf(fp,"        int16_t j = i + %ld;\n",k+1);
    fprintf(fp,"        middle = D0[i] ^ D1[i] ^ D0[j];\n");
    fprintf(fp,"        Out[i] = D0[i];\n");
    fprintf(fp,"        Out[j] = D3[i] ^ middle;\n");
    fprintf(fp,"        Out[j + %ld] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;\n",k+1);
    fprintf(fp,"        middle = D1[j] ^ D2[i] ^ D2[j];\n");
    fprintf(fp,"        Out[j + %ld] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;\n",2*(k+1));
    fprintf(fp,"        Out[i + %ld] = D5[j] ^ middle;\n",4*(k+1));
    fprintf(fp,"        Out[j + %ld] = D2[j];\n",4*(k+1));
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int16_t i = %ld; i < %ld; i++)\n",k-1,k+1);
    fprintf(fp,"    {\n");
    fprintf(fp,"        int16_t j = i + %ld;\n",k+1);
    fprintf(fp,"        middle = D0[i] ^ D1[i] ^ D0[j];\n");
    fprintf(fp,"        Out[i] = D0[i];\n");
    fprintf(fp,"        Out[j] = D3[i] ^ middle;\n");
    fprintf(fp,"        Out[j + %ld] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;\n",k+1);
    fprintf(fp,"        middle = D1[j] ^ D2[i];\n");
    fprintf(fp,"        Out[j + %ld] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;\n",2*(k+1));
    fprintf(fp,"        Out[i + %ld] = D5[j] ^ middle;\n",4*(k+1));
    fprintf(fp,"    }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}

void fprint_karat5_5k(FILE * fp, char * func_name, uint64_t len){
    uint64_t k=len/5;
    fprintf(fp,"static inline void %s(poly8x16_t *Out,   poly8x16_t *A,   poly8x16_t *B){\n",func_name);
    fprintf(fp,"    static poly8x16_t *a0, *b0, *a1, *b1, *a2, *b2, * a3, * b3, *a4, *b4;\n");
    fprintf(fp,"    static poly8x16_t aa01[%ld], bb01[%ld], aa02[%ld], bb02[%ld], aa03[%ld], bb03[%ld],\n",k,k,k,k,k,k);
    fprintf(fp,"           aa04[%ld], bb04[%ld], aa12[%ld], bb12[%ld], aa13[%ld], bb13[%ld],\n",k,k,k,k,k,k);
    fprintf(fp,"           aa14[%ld], bb14[%ld], aa23[%ld], bb23[%ld], aa24[%ld], bb24[%ld], aa34[%ld], bb34[%ld];\n",k,k,k,k,k,k,k,k);
    fprintf(fp,"    static poly8x16_t D0[%ld], D1[%ld], D2[%ld], D3[%ld], D4[%ld],\n",2*k,2*k,2*k,2*k,2*k);
    fprintf(fp,"           D01[%ld], D02[%ld], D03[%ld], D04[%ld], D12[%ld],\n",2*k,2*k,2*k,2*k,2*k);
    fprintf(fp,"           D13[%ld], D14[%ld], D23[%ld], D24[%ld], D34[%ld];\n",2*k,2*k,2*k,2*k,2*k);
    fprintf(fp,"\n");
    
    fprintf(fp,"   a0 = A;\n");
    fprintf(fp,"   a1 = a0 + %ld;\n",k);
    fprintf(fp,"   a2 = a1 + %ld;\n",k);
    fprintf(fp,"   a3 = a2 + %ld;\n",k);
    fprintf(fp,"   a4 = a3 + %ld;\n",k);
    fprintf(fp,"\n");

    fprintf(fp,"   b0 = B;\n");
    fprintf(fp,"   b1 = b0 + %ld;\n",k);
    fprintf(fp,"   b2 = b1 + %ld;\n",k);
    fprintf(fp,"   b3 = b2 + %ld;\n",k);
    fprintf(fp,"   b4 = b3 + %ld;\n",k);
    fprintf(fp,"\n");
    
    fprintf(fp,"   for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"       aa01[i] = a0[i] ^ a1[i];\n");
    fprintf(fp,"       bb01[i] = b0[i] ^ b1[i];\n");
    fprintf(fp,"       aa02[i] = a0[i] ^ a2[i];\n");
    fprintf(fp,"       bb02[i] = b0[i] ^ b2[i];\n");
    fprintf(fp,"       aa03[i] = a0[i] ^ a3[i];\n");
    fprintf(fp,"       bb03[i] = b0[i] ^ b3[i];\n");
    fprintf(fp,"       aa04[i] = a0[i] ^ a4[i];\n");
    fprintf(fp,"       bb04[i] = b0[i] ^ b4[i];\n");
    fprintf(fp,"       aa12[i] = a1[i] ^ a2[i];\n");
    fprintf(fp,"       bb12[i] = b1[i] ^ b2[i];\n");
    fprintf(fp,"       aa13[i] = a1[i] ^ a3[i];\n");
    fprintf(fp,"       bb13[i] = b1[i] ^ b3[i];\n");
    fprintf(fp,"       aa14[i] = a1[i] ^ a4[i];\n");
    fprintf(fp,"       bb14[i] = b1[i] ^ b4[i];\n");
    fprintf(fp,"       aa23[i] = a2[i] ^ a3[i];\n");
    fprintf(fp,"       bb23[i] = b2[i] ^ b3[i];\n");
    fprintf(fp,"       aa24[i] = a2[i] ^ a4[i];\n");
    fprintf(fp,"       bb24[i] = b2[i] ^ b4[i];\n");
    fprintf(fp,"       aa34[i] = a3[i] ^ a4[i];\n");
    fprintf(fp,"       bb34[i] = b3[i] ^ b4[i];\n");
    fprintf(fp,"   }\n");
    fprintf(fp,"\n");
    
    fprintf(fp,"   gfmul_%ld(D01, aa01, bb01);\n",k);
    fprintf(fp,"   gfmul_%ld(D02, aa02, bb02);\n",k);
    fprintf(fp,"   gfmul_%ld(D03, aa03, bb03);\n",k);
    fprintf(fp,"   gfmul_%ld(D04, aa04, bb04);\n",k);
    fprintf(fp,"\n");
    
    fprintf(fp,"   gfmul_%ld(D12, aa12, bb12);\n",k);
    fprintf(fp,"   gfmul_%ld(D13, aa13, bb13);\n",k);
    fprintf(fp,"   gfmul_%ld(D14, aa14, bb14);\n",k);
    fprintf(fp,"\n");
    
    fprintf(fp,"   gfmul_%ld(D23, aa23, bb23);\n",k);
    fprintf(fp,"   gfmul_%ld(D24, aa24, bb24);\n",k);
    fprintf(fp,"\n");
    
    fprintf(fp,"   gfmul_%ld(D34, aa34, bb34);\n",k);
    fprintf(fp,"\n");

    fprintf(fp,"   gfmul_%ld(D0, a0, b0);\n",k);
    fprintf(fp,"   gfmul_%ld(D1, a1, b1);\n",k);
    fprintf(fp,"   gfmul_%ld(D2, a2, b2);\n",k);
    fprintf(fp,"   gfmul_%ld(D3, a3, b3);\n",k);
    fprintf(fp,"   gfmul_%ld(D4, a4, b4);\n",k);
    fprintf(fp,"\n");
    
    fprintf(fp,"   for (int16_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"       int16_t j = i + %ld;\n",k);
    fprintf(fp,"       Out[i] = D0[i];\n");
    fprintf(fp,"       Out[i + %ld] = D0[j] ^ D01[i] ^ D0[i] ^ D1[i];\n",k);
    fprintf(fp,"       Out[i + %ld] = D1[i] ^ D02[i] ^ D0[i] ^ D2[i] ^ D01[j] ^ D0[j] ^ D1[j];\n",2*k);
    fprintf(fp,"       Out[i + %ld] = D1[j] ^ D03[i] ^ D0[i] ^ D3[i] ^ D12[i] ^ D1[i] ^ D2[i] ^ D02[j] ^ D0[j] ^ D2[j];\n",3*k);
    fprintf(fp,"       Out[i + %ld] = D2[i] ^ D04[i] ^ D0[i] ^ D4[i] ^ D13[i] ^ D1[i] ^ D3[i] ^ D03[j] ^ D0[j] ^ D3[j] ^ D12[j] ^ D1[j] ^ D2[j];\n",4*k);
    fprintf(fp,"       Out[i + %ld] = D2[j] ^ D14[i] ^ D1[i] ^ D4[i] ^ D23[i] ^ D2[i] ^ D3[i] ^ D04[j] ^ D0[j] ^ D4[j] ^ D13[j] ^ D1[j] ^ D3[j];\n",5*k);
    fprintf(fp,"       Out[i + %ld] = D3[i] ^ D24[i] ^ D2[i] ^ D4[i] ^ D14[j] ^ D1[j] ^ D4[j] ^ D23[j] ^ D2[j] ^ D3[j];\n",6*k);
    fprintf(fp,"       Out[i + %ld] = D3[j] ^ D34[i] ^ D3[i] ^ D4[i] ^ D24[j] ^ D2[j] ^ D4[j];\n",7*k);
    fprintf(fp,"       Out[i + %ld] = D4[i] ^ D34[j] ^ D3[j] ^ D4[j];\n",8*k);
    fprintf(fp,"       Out[i + %ld] = D4[j];\n",9*k);
    fprintf(fp,"   }\n");
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}

void fprint_TC3_128_3k(FILE * fp, char * func_name, uint64_t len){
    uint64_t k=len/3;
    fprintf(fp,"static inline void %s(poly8x16_t *Out,   poly8x16_t *A256,   poly8x16_t *B256){\n",func_name);
    fprintf(fp,"    static poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp,"    static poly8x16_t W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld], tmp[%ld];\n",2*k,2*k+3,2*k+4,2*k+4,2*k,2*k+4);
    fprintf(fp,"    static poly8x16_t zero = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};\n");
    fprintf(fp,"\n");

    fprintf(fp,"    U0 = (poly8x16_t *)&A256[0];\n");
    fprintf(fp,"    U1 = (poly8x16_t *)&A256[%ld];\n",k);
    fprintf(fp,"    U2 = (poly8x16_t *)&A256[%ld];\n",2*k);
    fprintf(fp,"    V0 = (poly8x16_t *)&B256[0];\n");
    fprintf(fp,"    V1 = (poly8x16_t *)&B256[%ld];\n",k);
    fprintf(fp,"    V2 = (poly8x16_t *)&B256[%ld];\n",2*k);
    fprintf(fp,"\n");
    
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");
    
    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",k);
    fprintf(fp,"\n");
    
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
    fprintf(fp,"\n");
    
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = W0[%ld];\n",k,k);
    fprintf(fp,"    W3[%ld] = W0[%ld];\n",k+1,k+1);
    fprintf(fp,"    W2[%ld] = W4[%ld];\n",k,k);
    fprintf(fp,"    W2[%ld] = W4[%ld];\n",k+1,k+1);        
    fprintf(fp,"\n");
    
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");
    
    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",k+2);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+4);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");
    
    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",k+2);
    fprintf(fp,"\n");
    
    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp,"\n");

    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",k);
    fprintf(fp,"\n");
    
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+4);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");
    
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W2[i] = W2[i1] ^ W0[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k-1,2*k);
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k,2*k+1);
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k+1,2*k+2);
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k+2,2*k+3);
    fprintf(fp,"\n");

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
    fprintf(fp,"    divide_by_x_plus_one_128(tmp, W2, %ld);\n",2*k+4);
    fprintf(fp,"\n");
    
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        tmp[i] = W3[i1] ^ W1[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k-1,2*k);
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k,2*k+1);
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+1,2*k+2);
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+2,2*k+3);
    fprintf(fp,"    divide_by_x_plus_one_128(tmp, W3, %ld);\n",2*k+3);
    fprintf(fp,"\n");
    
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k,2*k+3);
    fprintf(fp,"        W1[i] = W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");
    
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");
    
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
void fprint_TC3_128_3kp2(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp, "static inline void %s(poly8x16_t *Out,   poly8x16_t *A256,   poly8x16_t *B256){\n",func_name);
    uint64_t k=len/3;
    fprintf(fp, "    static poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp, "    static poly8x16_t W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld], tmp[%ld];\n",2*k+2,2*k+3,2*k+4,2*k+4,2*k,2*k+4);
    fprintf(fp, "    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};\n");
    fprintf(fp, "\n");

    fprintf(fp, "    U0 = (poly8x16_t *)&A256[0];\n");
    fprintf(fp, "    U1 = (poly8x16_t *)&A256[%ld];\n",(k+1));
    fprintf(fp, "    U2 = (poly8x16_t *)&A256[%ld];\n",2*k+2);
    fprintf(fp, "    V0 = (poly8x16_t *)&B256[0];\n");
    fprintf(fp, "    V1 = (poly8x16_t *)&B256[%ld];\n",(k+1));
    fprintf(fp, "    V2 = (poly8x16_t *)&B256[%ld];\n",2*k+2);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp, "        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp, "        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k,k,k);
    fprintf(fp, "    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k,k,k);
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(W1, W2, W3);\n",(k+1));
    fprintf(fp, "\n");

    fprintf(fp, "    W0[0] = zero;\n");
    fprintf(fp, "    W4[0] = zero;\n");
    fprintf(fp, "    W0[1] = U1[0];\n");
    fprintf(fp, "    W4[1] = V1[0];\n");
    fprintf(fp, "    for (int32_t i = 1 ; i < %ld ; i++) {\n",(k+1));
    fprintf(fp, "        W0[i + 1] = U1[i] ^ U2[i - 1];\n");
    fprintf(fp, "        W4[i + 1] = V1[i] ^ V2[i - 1];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",(k+1));
    fprintf(fp, "        W3[i] ^= W0[i];\n");
    fprintf(fp, "        W2[i] ^= W4[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W3[%ld] = W0[%ld];\n",(k+1),(k+1));
    fprintf(fp, "    W2[%ld] = W4[%ld];\n",(k+1),(k+1));
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",(k+1));
    fprintf(fp, "        W0[i] ^= U0[i];\n");
    fprintf(fp, "        W4[i] ^= V0[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(tmp, W3, W2);\n",k+2);
    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+4);
    fprintf(fp, "        W3[i] = tmp[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(W2, W0, W4);\n",k+2);
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(W0, U0, V0);\n",(k+1));
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+4);
    fprintf(fp, "        W3[i] ^= W2[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp, "        W1[i] ^= W0[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+1);
    fprintf(fp, "        int32_t i1 = i + 1;\n");
    fprintf(fp, "        W2[i] = W2[i1] ^ W0[i1];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W2[%ld] = W2[%ld];\n",2*k+1,2*k+2);
    fprintf(fp, "    W2[%ld] = W2[%ld];\n",2*k+2,2*k+3);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp, "        tmp[i] = W2[i] ^ W3[i] ^ W4[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k,2*k+3);
    fprintf(fp, "        tmp[i] = W2[i] ^ W3[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    tmp[%ld] = W3[%ld];\n",2*k+3,2*k+3);
    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp, "        tmp[i + 3] ^= W4[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    divide_by_x_plus_one_128(tmp, W2, %ld);\n",2*k+4);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+1);
    fprintf(fp, "        int32_t i1 = i + 1;\n");
    fprintf(fp, "        tmp[i] = W3[i1] ^ W1[i1];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    tmp[%ld] = W3[%ld];\n",2*k+1,2*k+2);
    fprintf(fp, "    tmp[%ld] = W3[%ld];\n",2*k+2,2*k+3);
    fprintf(fp, "    divide_by_x_plus_one_128(tmp, W3, %ld);\n",2*k+3);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp, "        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k,2*k+2);
    fprintf(fp, "        W1[i] ^= W2[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W1[%ld] = W2[%ld];\n",2*k+2,2*k+2);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp, "        W2[i] ^= W3[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0; i < %ld; i++) {\n",k-1);
    fprintf(fp, "        int32_t j = i + %ld;\n",(k+1));
    fprintf(fp, "        Out[i] = W0[i];\n");
    fprintf(fp, "        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp, "        Out[j + %ld] = W1[j] ^ W2[i];\n",(k+1));
    fprintf(fp, "        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k+2);
    fprintf(fp, "        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k+4);
    fprintf(fp, "        Out[j + %ld] = W4[j];\n",4*k+4);
    fprintf(fp, "    }\n");        
    fprintf(fp, "    for (int32_t i = %ld; i < %ld; i++) {\n",k-1,(k+1));
    fprintf(fp, "        int32_t j = i + %ld;\n",(k+1));
    fprintf(fp, "        Out[i] = W0[i];\n");
    fprintf(fp, "        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp, "        Out[j + %ld] = W1[j] ^ W2[i];\n",(k+1));
    fprintf(fp, "        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k+2);
    fprintf(fp, "        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k+4);
    fprintf(fp, "    }\n");
    fprintf(fp, "    Out[%ld] ^= W1[%ld];\n",3*k+3,2*k+2);
    fprintf(fp, "    Out[%ld] ^= W2[%ld];\n",4*k+4,2*k+2);
    fprintf(fp, "}\n");
    fprintf(fp,"\n");
}
void fprint_TC3_128_3kp1(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp,"static inline void %s(poly8x16_t *Out,   poly8x16_t *A256,   poly8x16_t *B256){\n",func_name);
    uint64_t k=len/3;
    fprintf(fp,"    static poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp,"    static poly8x16_t W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld], tmp[%ld];\n",2*k+2,2*k+3,2*k+4,2*k+4,2*k-2,2*k+4);
    fprintf(fp,"    static poly8x16_t zero = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};\n");
    fprintf(fp,"\n");

    fprintf(fp,"    U0 = (poly8x16_t *)&A256[0];\n");
    fprintf(fp,"    U1 = (poly8x16_t *)&A256[%ld];\n",(k+1));
    fprintf(fp,"    U2 = (poly8x16_t *)&A256[%ld];\n",2*k+2);
    fprintf(fp,"    V0 = (poly8x16_t *)&B256[0];\n");
    fprintf(fp,"    V1 = (poly8x16_t *)&B256[%ld];\n",(k+1));
    fprintf(fp,"    V2 = (poly8x16_t *)&B256[%ld];\n",2*k+2);
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k-1);
    fprintf(fp,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k-1,k-1,k-1);
    fprintf(fp,"    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k-1,k-1,k-1);
    fprintf(fp,"    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k,k,k);
    fprintf(fp,"    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k,k,k);
    fprintf(fp,"\n");

    fprintf(fp,"    gfmul_%ld(W1, W2, W3);\n",(k+1));
    fprintf(fp,"\n");

    fprintf(fp,"    W0[0] = zero;\n");
    fprintf(fp,"    W4[0] = zero;\n");
    fprintf(fp,"    W0[1] = U1[0];\n");
    fprintf(fp,"    W4[1] = V1[0];\n");
    fprintf(fp,"    for (int32_t i = 1 ; i < %ld ; i++) {\n",k);
    fprintf(fp,"        W0[i + 1] = U1[i] ^ U2[i - 1];\n");
    fprintf(fp,"        W4[i + 1] = V1[i] ^ V2[i - 1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W0[%ld] = U1[%ld];\n",(k+1),k);
    fprintf(fp,"    W4[%ld] = V1[%ld];\n",(k+1),k);
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",(k+1));
    fprintf(fp,"        W3[i] ^= W0[i];\n");
    fprintf(fp,"        W2[i] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W3[%ld] = W0[%ld];\n",(k+1),(k+1));
    fprintf(fp,"    W2[%ld] = W4[%ld];\n",(k+1),(k+1));
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",(k+1));
    fprintf(fp,"        W0[i] ^= U0[i];\n");
    fprintf(fp,"        W4[i] ^= V0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");

    fprintf(fp,"    gfmul_%ld(tmp, W3, W2);\n",(k+1)+1);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+4);
    fprintf(fp,"        W3[i] = tmp[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");

    fprintf(fp,"    gfmul_%ld(W2, W0, W4);\n",(k+1)+1);
    fprintf(fp,"\n");

    fprintf(fp,"    gfmul_%ld(W4, U2, V2);\n",k-1);
    fprintf(fp,"\n");

    fprintf(fp,"    gfmul_%ld(W0, U0, V0);\n",(k+1));
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+4);
    fprintf(fp,"        W3[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp,"        W1[i] ^= W0[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        W2[i] = W2[i1] ^ W0[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k+1,2*k+2);
    fprintf(fp,"    W2[%ld] = W2[%ld];\n",2*k+2,2*k+3);
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-2);
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k-2,2*k+3);
    fprintf(fp,"        tmp[i] = W2[i] ^ W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+3,2*k+3);
    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-2);
    fprintf(fp,"        tmp[i + 3] ^= W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    divide_by_x_plus_one_128(tmp, W2, %ld);\n",2*k+4);
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+1);
    fprintf(fp,"        int32_t i1 = i + 1;\n");
    fprintf(fp,"        tmp[i] = W3[i1] ^ W1[i1];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+1,2*k+2);
    fprintf(fp,"    tmp[%ld] = W3[%ld];\n",2*k+2,2*k+3);
    fprintf(fp,"    divide_by_x_plus_one_128(tmp, W3, %ld);\n",2*k+3);
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k-2);
    fprintf(fp,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    for (int32_t i = %ld ; i < %ld ; i++) {\n",2*k-2,2*k+2);
    fprintf(fp,"        W1[i] ^= W2[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    W1[%ld] = W2[%ld];\n",2*k+2,2*k+2);
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp,"        W2[i] ^= W3[i];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"\n");

    fprintf(fp,"    for (int32_t i = 0; i < %ld; i++) {\n",k-3);
    fprintf(fp,"        int32_t j = i + %ld;\n",(k+1));
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",(k+1));
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k+2);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k+4);
    fprintf(fp,"        Out[j + %ld] = W4[j];\n",4*k+4);
    fprintf(fp,"    }\n");        
    fprintf(fp,"    for (int32_t i = %ld; i < %ld; i++) {\n",k-3,(k+1));
    fprintf(fp,"        int32_t j = i + %ld;\n",(k+1));
    fprintf(fp,"        Out[i] = W0[i];\n");
    fprintf(fp,"        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp,"        Out[j + %ld] = W1[j] ^ W2[i];\n",(k+1));
    fprintf(fp,"        Out[j + %ld] = W2[j] ^ W3[i];\n",2*k+2);
    fprintf(fp,"        Out[i + %ld] = W3[j] ^ W4[i];\n",4*k+4);
    fprintf(fp,"    }\n");
    fprintf(fp,"    Out[%ld] ^= W1[%ld];\n",3*k+3,2*k+2);
    fprintf(fp,"    Out[%ld] ^= W2[%ld];\n",4*k+4,2*k+2);
    fprintf(fp,"}\n");
    fprintf(fp,"\n");
}

void fprint_TC3_64_3k(FILE * fp, char * func_name, uint64_t len){
    //사실 64 
    fprintf(fp, "static inline void %s(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){\n",func_name);
    
    uint64_t k=len/3;

    fprintf(fp, "    poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp, "    static poly8x16_t W0[%ld], W1[%ld], W4[%ld];\n",k*2,k*2+2,k*2);
    fprintf(fp, "    static poly8x16_t W2[%ld], W3[%ld], tmp[%ld];\n",k*2+2,k*2+2,k*2+2);
    fprintf(fp, "    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};\n");
    fprintf(fp, "\n");
    
    fprintf(fp, "    U0 = (poly8x16_t *)&A128[0];\n");
    fprintf(fp, "    U1 = (poly8x16_t *)&A128[%ld];\n",k);
    fprintf(fp, "    U2 = (poly8x16_t *)&A128[%ld];\n",k*2);
    fprintf(fp, "    V0 = (poly8x16_t *)&B128[0];\n");
    fprintf(fp, "    V1 = (poly8x16_t *)&B128[%ld];\n",k);
    fprintf(fp, "    V2 = (poly8x16_t *)&B128[%ld];\n",k*2);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp, "        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp, "        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(W1, W2, W3);\n",k);
    fprintf(fp, "\n");

    fprintf(fp, "    uint64_t *U1_64 = ((uint64_t *) U1);\n");
    fprintf(fp, "    uint64_t *V1_64 = ((uint64_t *) V1);\n");
    fprintf(fp, "    W0[0] = vcombine_p8(zero,(poly8x8_t)U1_64[0]);\n");
    fprintf(fp, "    W4[0] = vcombine_p8(zero,(poly8x8_t)V1_64[0]);\n");
    fprintf(fp, "    U1_64 = ((uint64_t *) U1) + 1;\n");
    fprintf(fp, "    V1_64 = ((uint64_t *) V1) + 1;\n");
    fprintf(fp, "    for(int32_t i = 0; i < %ld; i++) {\n",k-1);
    fprintf(fp, "        int32_t i2 = i << 1;\n");
    fprintf(fp, "        int32_t i1 = i + 1;\n");
    fprintf(fp, "        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));\n");
    fprintf(fp, "        W0[i1] ^= U2[i];\n");
    fprintf(fp, "        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));\n");
    fprintf(fp, "        W4[i1] ^= V2[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W0[%ld] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld],zero) ^ U2[%ld];\n",k,k*2-2,k-1);
    fprintf(fp, "    W4[%ld] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[%ld],zero) ^ V2[%ld];\n",k,k*2-2,k-1);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp, "        W3[i] ^= W0[i];\n");
    fprintf(fp, "        W2[i] ^= W4[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W3[%ld] = W0[%ld];\n",k,k);
    fprintf(fp, "    W2[%ld] = W4[%ld];\n",k,k);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp, "        W0[i] ^= U0[i];\n");
    fprintf(fp, "        W4[i] ^= V0[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(tmp, W3, W2);\n",k+1);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld; i++) {\n",k*2 + 2);
    fprintf(fp, "        W3[i] = tmp[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(W2, W0, W4);\n",k+1);
    fprintf(fp, "    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp, "    gfmul_%ld(W0, U0, V0);\n",k);
    fprintf(fp, "\n");


    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2 + 2);
    fprintf(fp, "        W3[i] ^= W2[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2);
    fprintf(fp, "        W1[i] ^= W0[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    U1_64 = ((uint64_t *) W2) + 1;\n");
    fprintf(fp, "    uint64_t * U2_64 = ((uint64_t *) W0) + 1;\n");
    fprintf(fp, "    for(int32_t i = 0; i < %ld; i++) {\n",2*k-1);
    fprintf(fp, "        int32_t i2 = i << 1;\n");
    fprintf(fp, "        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));\n");
    fprintf(fp, "        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W2[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[%ld],zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld])));\n",2*k-1,4*k-2,4*k-2);
    fprintf(fp, "    W2[%ld]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld])));\n",2*k,4*k);
    fprintf(fp, "    W2[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld],zero);\n",2*k+1,4*k+2);
    fprintf(fp, "\n");

    fprintf(fp, "    U1_64 = ((uint64_t *) W4);\n");
    fprintf(fp, "    tmp[0] = W2[0] ^ W3[0] ^ W4[0];\n");
    fprintf(fp, "    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);\n");
    fprintf(fp, "    U1_64 = ((uint64_t *) W4) + 1;\n");
    fprintf(fp, "    for(int32_t i = 2; i < %ld; i++) {\n",2*k);
    fprintf(fp, "        int32_t i2 = i << 1;\n");
    fprintf(fp, "        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld])));\n",2*k,2*k,2*k,4*k-4);
    fprintf(fp, "    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld], zero);\n",2*k+1,2*k+1,2*k+1,4*k-2);
    fprintf(fp, "    divide_by_x_plus_one_64(W2, tmp, %ld);\n",k*4 + 4);
    fprintf(fp, "\n");

    fprintf(fp, "    U1_64 = ((uint64_t *) W3) + 1;\n");
    fprintf(fp, "    U2_64 = ((uint64_t *) W1) + 1;\n");    
    fprintf(fp, "    for(int32_t i = 0; i < %ld; i++) {\n",k*2 - 1);
    fprintf(fp, "        int32_t i2 = i << 1;\n");
    fprintf(fp, "        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    tmp[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[%ld], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld])));\n",2*k-1,4*k-2,4*k-2);
    fprintf(fp, "    tmp[%ld]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld])));\n",2*k,k*4);
    fprintf(fp, "    tmp[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld], zero);\n",2*k+1,k*4+2);
    fprintf(fp, "    divide_by_x_plus_one_64(W3, tmp, %ld);\n",4*k+3);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp, "        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W1[%ld] = W2[%ld];\n",2*k,2*k);
    fprintf(fp, "    W1[%ld] = W2[%ld];\n",2*k+1,2*k+1);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+1);
    fprintf(fp, "        W2[i] ^= W3[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");
    
    fprintf(fp, "    for (int32_t i = 0; i < %ld; i++)\n",k);
    fprintf(fp, "    {\n");
    fprintf(fp, "        int32_t j = i + %ld;\n",k);
    fprintf(fp, "        Out[i] = W0[i];\n");
    fprintf(fp, "        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp, "        Out[j + %ld] = W1[j] ^ W2[i];\n",k);
    fprintf(fp, "        Out[j + %ld] = W2[j] ^ W3[i];\n",k*2);
    fprintf(fp, "        Out[i + %ld] = W3[j] ^ W4[i];\n",k*4);
    fprintf(fp, "        Out[j + %ld] = W4[j];\n",k*4);
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    Out[%ld] ^= W1[%ld];\n",3*k,2*k);
    fprintf(fp, "    Out[%ld] ^= W1[%ld];\n",3*k+1,2*k+1);
    fprintf(fp, "    Out[%ld] ^= W2[%ld];\n",4*k,2*k);
    fprintf(fp, "    Out[%ld] ^= W2[%ld];\n",4*k+1,2*k+1);
    fprintf(fp, "    Out[%ld] ^= W3[%ld];\n",5*k,2*k);
    fprintf(fp, "}\n");
    fprintf(fp, "\n");
}
void fprint_TC3_64_3kp1(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp, "static inline void %s(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){\n",func_name);
    uint64_t k=len/3;

    fprintf(fp ,"    poly8x16_t U0[%ld], U1[%ld], U2[%ld], V0[%ld], V1[%ld], V2[%ld];\n",k+1,k+1,k,k+1,k+1,k);
    fprintf(fp ,"    poly8x16_t W0[%ld], W1[%ld], W2[%ld], W3[%ld], W4[%ld];\n",k*2+2,k*2+2,k*2+2,k*2+3,k*2);
    fprintf(fp ,"    poly8x16_t tmp[%ld];\n",k*2+3);
    fprintf(fp ,"    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    uint64_t *A = (uint64_t *) A128;\n");
    fprintf(fp ,"    uint64_t *B = (uint64_t *) B128;\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
    fprintf(fp ,"        int32_t i2 = i << 1;\n");
    fprintf(fp ,"        U0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2])));\n");
    fprintf(fp ,"        V0[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2])));\n");
    fprintf(fp ,"        U1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + %ld])));\n",k*2 + 1);
    fprintf(fp ,"        V1[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + %ld])));\n",k*2 + 1);
    fprintf(fp ,"        U2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& A[i2 + %ld])));\n",k*4 + 2);
    fprintf(fp ,"        V2[i]= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& B[i2 + %ld])));\n",k*4 + 2);
    fprintf(fp ,"    }\n");
    fprintf(fp ,"    U0[%ld]= (poly8x16_t)vcombine_p8((poly8x8_t)A[%ld], zero);\n",k,2*k);
    fprintf(fp ,"    V0[%ld]= (poly8x16_t)vcombine_p8((poly8x8_t)B[%ld], zero);\n",k,2*k);
    fprintf(fp ,"    U1[%ld]= (poly8x16_t)vcombine_p8((poly8x8_t)A[%ld], zero);\n",k,k*4 + 1);
    fprintf(fp ,"    V1[%ld]= (poly8x16_t)vcombine_p8((poly8x8_t)B[%ld], zero);\n",k,k*4 + 1);
    fprintf(fp, "\n");

    fprintf(fp ,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp ,"        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp ,"        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp ,"    }\n");
    fprintf(fp ,"    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k,k,k);
    fprintf(fp ,"    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k,k,k);
    fprintf(fp, "\n");

    fprintf(fp ,"    gfmul_%ld(W1, W2, W3);\n",k+1);
    fprintf(fp, "\n");

    fprintf(fp ,"    uint64_t *U1_64 = ((uint64_t *) U1);\n");
    fprintf(fp ,"    uint64_t *V1_64 = ((uint64_t *) V1);\n");
    fprintf(fp ,"    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);\n");
    fprintf(fp ,"    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    U1_64 = ((uint64_t *) U1) + 1;\n");
    fprintf(fp ,"    V1_64 = ((uint64_t *) V1) + 1;\n");
    fprintf(fp ,"    for(int32_t i = 0; i < %ld; i++) {\n",k);
    fprintf(fp ,"        int32_t i2 = i << 1;\n");
    fprintf(fp ,"        int32_t i1 = i + 1;\n");
    fprintf(fp ,"        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));\n");
    fprintf(fp ,"        W0[i1] ^= U2[i];\n");
    fprintf(fp ,"        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));\n");
    fprintf(fp ,"        W4[i1] ^= V2[i];\n");
    fprintf(fp ,"    }\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k+1);
    fprintf(fp ,"        W3[i] ^= W0[i];\n");
    fprintf(fp ,"        W2[i] ^= W4[i];\n");
    fprintf(fp ,"    }\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k+1);
    fprintf(fp ,"        W0[i] ^= U0[i];\n");
    fprintf(fp ,"        W4[i] ^= V0[i];\n");
    fprintf(fp ,"    }\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    gfmul_%ld(tmp, W3, W2);\n",k+1);
    fprintf(fp, "\n");

    fprintf(fp ,"    for (int32_t i = 0 ; i < %ld; i++) {\n",k*2 + 2);
    fprintf(fp ,"        W3[i] = tmp[i];\n");
    fprintf(fp ,"    }\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    gfmul_%ld(W2, W0, W4);\n",k+1);
    fprintf(fp ,"    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp ,"    gfmul_%ld(W0, U0, V0);\n",k+1);
    fprintf(fp, "\n");

    fprintf(fp ,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2 + 2);
    fprintf(fp ,"        W3[i] ^= W2[i];\n");
    fprintf(fp ,"    }\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2+1);
    fprintf(fp ,"        W1[i] ^= W0[i];\n");
    fprintf(fp ,"    }\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    U1_64 = ((uint64_t *) W2) + 1;\n");
    fprintf(fp ,"    uint64_t * U2_64 = ((uint64_t *) W0) + 1;\n");
    fprintf(fp ,"    for(int32_t i = 0; i < %ld; i++) {\n",2*k);
    fprintf(fp ,"        int32_t i2 = i << 1;\n");
    fprintf(fp ,"        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));\n");
    fprintf(fp ,"        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));\n");
    fprintf(fp ,"    }\n");
    fprintf(fp ,"    W2[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[%ld], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld])));\n",k*2,k*4,k*4);
    fprintf(fp ,"    W2[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld], zero);\n",k*2 + 1, k*4+2);
    fprintf(fp, "\n");

    fprintf(fp ,"    U1_64 = ((uint64_t *) W4);\n");
    fprintf(fp ,"    tmp[0] = W2[0] ^ W3[0] ^ W4[0];\n");
    fprintf(fp ,"    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    U1_64 = ((uint64_t *) W4) + 1;\n");
    fprintf(fp ,"    for(int32_t i = 2; i < %ld; i++) {\n",2*k);
    fprintf(fp ,"        int32_t i2 = i << 1;\n");
    fprintf(fp ,"        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));\n");
    fprintf(fp ,"    }\n");
    fprintf(fp ,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld])));\n",2*k,2*k,2*k,4*k-4);
    fprintf(fp ,"    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld], zero);\n",2*k+1,2*k+1,2*k+1,4*k-2);
    fprintf(fp ,"    divide_by_x_plus_one_64(W2, tmp, %ld);\n",k*4 + 4);
    fprintf(fp, "\n");

    fprintf(fp ,"    U1_64 = ((uint64_t *) W3) + 1;\n");
    fprintf(fp ,"    U2_64 = ((uint64_t *) W1) + 1;\n");
    fprintf(fp ,"    for(int32_t i = 0; i < %ld; i++) {\n",k*2);
    fprintf(fp ,"        int32_t i2 = i << 1;\n");
    fprintf(fp ,"        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));\n");
    fprintf(fp ,"    }\n");
    fprintf(fp ,"    tmp[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U2_64[%ld], zero) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld])));\n",k*2, k*4, k*4);
    fprintf(fp ,"    tmp[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld], zero);\n",2*k+1,k*4+2);
    fprintf(fp ,"    divide_by_x_plus_one_64(W3, tmp, %ld);\n",4*k+3);
    fprintf(fp, "\n");

    fprintf(fp ,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp ,"        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp ,"    }\n");
    fprintf(fp ,"    W1[%ld] ^= W2[%ld];\n",2*k,2*k);
    fprintf(fp ,"    W1[%ld] = W2[%ld];\n",2*k+1,2*k+1);
    fprintf(fp, "\n");

    fprintf(fp ,"    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+1);
    fprintf(fp ,"        W2[i] ^= W3[i];\n");
    fprintf(fp ,"    }\n");
    fprintf(fp, "\n");

    fprintf(fp ,"    for(int32_t i = 0; i < %ld; i++) {\n",k*2);
    fprintf(fp ,"        Out[i] = W0[i];\n");
    fprintf(fp ,"        Out[i + %ld] = W2[i];\n",k*2 + 1);
    fprintf(fp ,"        Out[i + %ld] = W4[i];\n",k*4 + 2);
    fprintf(fp ,"    }\n");
    fprintf(fp ,"    Out[%ld] = W0[%ld];\n",k*2,k*2);
    fprintf(fp ,"    Out[%ld] = W2[%ld];\n",k*4+1,k*2);
    fprintf(fp ,"    Out[%ld] ^= W2[%ld];\n",k*4+2,k*2+1);
    fprintf(fp, "\n");

    fprintf(fp ,"    U1_64 = ((uint64_t *) &Out[%ld]) + 1;\n",k);
    fprintf(fp ,"    U2_64 = ((uint64_t *) &Out[%ld]) + 1;\n",k*3+1);
    fprintf(fp ,"    poly8x16_t aux;\n");    
    fprintf(fp ,"    for(int32_t i = 0; i < %ld; i++) {\n",2*k+2);
    fprintf(fp ,"        int32_t i2 = i << 1;\n");
    fprintf(fp ,"        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ W1[i];\n");
    fprintf(fp ,"        vstrq_p128((poly128_t *) (& U1_64[i2]), vreinterpretq_p128_p8(aux));\n");
    fprintf(fp ,"        aux = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2]))) ^ W3[i];\n");
    fprintf(fp ,"        vstrq_p128((poly128_t *) (& U2_64[i2]), vreinterpretq_p128_p8(aux));\n");
    fprintf(fp ,"    }\n");
    fprintf(fp ,"}\n");
    fprintf(fp ,"\n");
}
void fprint_TC3_64_3kp2(FILE * fp, char * func_name, uint64_t len){
    fprintf(fp, "static inline void %s(poly8x16_t *Out,   poly8x16_t *A128,   poly8x16_t *B128){\n",func_name);
    uint64_t k=len/3;      
    fprintf(fp, "    poly8x16_t *U0, *U1, *U2, *V0, *V1, *V2;\n");
    fprintf(fp, "    static poly8x16_t W0[%ld], W1[%ld], W4[%ld];\n",k*2+2,k*2+3,k*2);
    fprintf(fp, "    static poly8x16_t W2[%ld], W3[%ld], tmp[%ld];\n",k*2+4,k*2+3,k*2+4);
    fprintf(fp, "    static poly8x8_t zero = {0, 0, 0, 0, 0, 0, 0, 0};\n");
    fprintf(fp, "\n");
    fprintf(fp, "    U0 = (poly8x16_t *)&A128[0];\n");
    fprintf(fp, "    U1 = (poly8x16_t *)&A128[%ld];\n",k+1);
    fprintf(fp, "    U2 = (poly8x16_t *)&A128[%ld];\n",k*2+2);
    fprintf(fp, "    V0 = (poly8x16_t *)&B128[0];\n");
    fprintf(fp, "    V1 = (poly8x16_t *)&B128[%ld];\n",k+1);
    fprintf(fp, "    V2 = (poly8x16_t *)&B128[%ld];\n",k*2+2);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k);
    fprintf(fp, "        W3[i] = U0[i] ^ U1[i] ^ U2[i];\n");
    fprintf(fp, "        W2[i] = V0[i] ^ V1[i] ^ V2[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");
    fprintf(fp, "    W3[%ld] = U0[%ld] ^ U1[%ld];\n",k,k,k);
    fprintf(fp, "    W2[%ld] = V0[%ld] ^ V1[%ld];\n",k,k,k);
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(W1, W2, W3);\n",k+1);
    fprintf(fp, "\n");

    fprintf(fp, "    uint64_t *U1_64 = ((uint64_t *) U1);\n");
    fprintf(fp, "    uint64_t *V1_64 = ((uint64_t *) V1);\n");
    fprintf(fp, "    W0[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);\n");
    fprintf(fp, "    W4[0] = (poly8x16_t)vcombine_p8(zero, (poly8x8_t)V1_64[0]);\n");
    fprintf(fp, "\n");

    fprintf(fp, "    U1_64 = ((uint64_t *) U1) + 1;\n");
    fprintf(fp, "    V1_64 = ((uint64_t *) V1) + 1;\n");
    fprintf(fp, "    for(int32_t i = 0; i < %ld; i++) {\n",k);
    fprintf(fp, "        int32_t i2 = i << 1;\n");
    fprintf(fp, "        int32_t i1 = i + 1;\n");
    fprintf(fp, "        W0[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));\n");
    fprintf(fp, "        W0[i1] ^= U2[i];\n");
    fprintf(fp, "        W4[i1] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& V1_64[i2])));\n");
    fprintf(fp, "        W4[i1] ^= V2[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    W0[%ld] = (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld], zero);\n",k+1,k*2);
    fprintf(fp, "    W4[%ld] = (poly8x16_t)vcombine_p8((poly8x8_t)V1_64[%ld], zero);\n",k+1,k*2);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k+1);
    fprintf(fp, "        W3[i] ^= W0[i];\n");
    fprintf(fp, "        W2[i] ^= W4[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W3[%ld] = W0[%ld];\n",k+1,k+1);
    fprintf(fp, "    W2[%ld] = W4[%ld];\n",k+1,k+1);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k+1);
    fprintf(fp, "        W0[i] ^= U0[i];\n");
    fprintf(fp, "        W4[i] ^= V0[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(tmp, W3, W2);\n",k+2);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld; i++) {\n",k*2 + 3);
    fprintf(fp, "        W3[i] = tmp[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    gfmul_%ld(W2, W0, W4);\n",k+2);
    fprintf(fp, "    gfmul_%ld(W4, U2, V2);\n",k);
    fprintf(fp, "    gfmul_%ld(W0, U0, V0);\n",k+1);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2 + 3);
    fprintf(fp, "        W3[i] ^= W2[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",k*2+2);
    fprintf(fp, "        W1[i] ^= W0[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    U1_64 = ((uint64_t *) W2) + 1;\n");
    fprintf(fp, "    uint64_t * U2_64 = ((uint64_t *) W0) + 1;\n");
    fprintf(fp, "\n");


    fprintf(fp, "    for(int32_t i = 0; i < %ld; i++) {\n",2*k+1);
    fprintf(fp, "        int32_t i2 = i << 1;\n");
    fprintf(fp, "        W2[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2])));\n");
    fprintf(fp, "        W2[i] ^= vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W2[%ld]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld]))) ^ (poly8x16_t)vcombine_p8((poly8x8_t)U2_64[%ld], zero);\n",2*k+1,4*k+2,4*k+2);
    fprintf(fp, "    W2[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld], zero);\n",2*k+2,4*k+4);
    fprintf(fp, "\n");

    fprintf(fp, "    U1_64 = ((uint64_t *) W4);\n");
    fprintf(fp, "    tmp[0] = W2[0] ^ W3[0] ^ W4[0];\n");
    fprintf(fp, "    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (poly8x16_t)vcombine_p8(zero, (poly8x8_t)U1_64[0]);\n");
    fprintf(fp, "\n");

    fprintf(fp, "    U1_64 = ((uint64_t *) W4) + 1;\n");
    fprintf(fp, "    for(int32_t i = 2; i < %ld; i++) {\n",2*k);
    fprintf(fp, "        int32_t i2 = i << 1;\n");
    fprintf(fp, "        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2-4])));\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld])));\n",2*k,2*k,2*k,4*k-4);
    fprintf(fp, "    tmp[%ld] = W2[%ld] ^ W3[%ld] ^ (poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld], zero);\n",2*k+1,2*k+1,2*k+1,4*k-2);
    fprintf(fp, "    tmp[%ld] = W2[%ld] ^ W3[%ld];\n",2*k+2,2*k+2,2*k+2);
    fprintf(fp, "    divide_by_x_plus_one_64(W2, tmp, %ld);\n",k*4 + 6);
    fprintf(fp, "\n");

    fprintf(fp, "    U1_64 = ((uint64_t *) W3) + 1;\n");
    fprintf(fp, "    U2_64 = ((uint64_t *) W1) + 1;\n");
    fprintf(fp, "    for(int32_t i = 0; i < %ld; i++) {\n",k*2 + 1);
    fprintf(fp, "        int32_t i2 = i << 1;\n");
    fprintf(fp, "        tmp[i] = vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[i2]))) ^ vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U2_64[i2])));\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    tmp[%ld]=vreinterpretq_p8_p128(vldrq_p128((poly128_t *)(& U1_64[%ld]))) ^ (poly8x16_t)vcombine_p8((poly8x8_t)U2_64[%ld], zero);\n",2*k+1,k*4+2,k*4+2);
    fprintf(fp, "    tmp[%ld]=(poly8x16_t)vcombine_p8((poly8x8_t)U1_64[%ld], zero);\n",2*k+2,k*4+4);
    fprintf(fp, "    divide_by_x_plus_one_64(W3, tmp, %ld);\n",4*k+5);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k);
    fprintf(fp, "        W1[i] ^= W2[i] ^ W4[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    W1[%ld] ^= W2[%ld];\n",2*k,2*k);
    fprintf(fp, "    W1[%ld] ^= W2[%ld];\n",2*k+1,2*k+1);
    fprintf(fp, "    W1[%ld] = W2[%ld];\n",2*k+2,2*k+2);
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = 0 ; i < %ld ; i++) {\n",2*k+2);
    fprintf(fp, "        W2[i] ^= W3[i];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");
    
    fprintf(fp, "    for (int32_t i = 0; i < %ld; i++)\n",k-1);
    fprintf(fp, "    {\n");
    fprintf(fp, "        int32_t j = i + %ld;\n",k+1);
    fprintf(fp, "        Out[i] = W0[i];\n");
    fprintf(fp, "        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp, "        Out[j + %ld] = W1[j] ^ W2[i];\n",k+1);
    fprintf(fp, "        Out[j + %ld] = W2[j] ^ W3[i];\n",k*2+2);
    fprintf(fp, "        Out[i + %ld] = W3[j] ^ W4[i];\n",k*4+4);
    fprintf(fp, "        Out[j + %ld] = W4[j];\n",k*4+4);
    fprintf(fp, "    }\n");
    fprintf(fp, "\n");

    fprintf(fp, "    for (int32_t i = %ld; i < %ld; i++)\n",k-1,k+1);
    fprintf(fp, "    {\n");
    fprintf(fp, "        int32_t j = i + %ld;\n",k+1);
    fprintf(fp, "        Out[i] = W0[i];\n");
    fprintf(fp, "        Out[j] = W0[j] ^ W1[i];\n");
    fprintf(fp, "        Out[j + %ld] = W1[j] ^ W2[i];\n",k+1);
    fprintf(fp, "        Out[j + %ld] = W2[j] ^ W3[i];\n",k*2+2);
    fprintf(fp, "        Out[i + %ld] = W3[j] ^ W4[i];\n",k*4+4);
    fprintf(fp, "    }\n");
    fprintf(fp, "    Out[%ld] ^= W1[%ld];\n",3*k+3,2*k+2);
    fprintf(fp, "    Out[%ld] ^= W2[%ld];\n",4*k+4,2*k+2);
    fprintf(fp, "}\n");
    fprintf(fp, "\n");
}

#if defined(BEST_ALG_NEON)
uint8_t best_algorithm[MAX_LEN+1] = BEST_ALG_NEON;
#else
uint8_t best_algorithm[MAX_LEN+1] = {0};
#endif

//env_num: 1=AVX2 + PCLMULQDQ, 2=AVX2 + VPCLMULQDQ, 3=NEON, 4=new
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
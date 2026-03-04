#ifndef _GFOPT_H_
#define _GFOPT_H_

#include <stdint.h>
#include <stdio.h>
#define N_TEST 100000

#ifdef ENV_AVX2
    #define WORD_LEN 256
    #ifdef ENV_VPCLMUL
        #define ENV_NUM 2
    #else
        #define ENV_NUM 1
    #endif
#endif
#ifdef ENV_NEON
    #define WORD_LEN 128
    #define ENV_NUM 3
#endif
#define MAX_N 57637
//#define MAX_LEN ((MAX_N/WORD_LEN)+((MAX_N%WORD_LEN==0)?0:1))
#define MAX_LEN 30 //100000/WORD_LEN


#define fprint_karat2(fp, function_name, len) {\
    if(len<2) printf("error (fprint_karat2) len = %d\n",len);\
    else if (len % 2 == 0) fprint_karat2_2k(fp, function_name, len);\
    else fprint_karat2_2kp1(fp, function_name, len);\
}
#define fprint_karat3(fp, function_name, len) {\
    if(len<3) printf("error (fprint_karat3) len = %d\n",len);\
    else if (len % 3 == 0) fprint_karat3_3k(fp, function_name, len);\
    else if (len % 3 == 1) fprint_karat3_3kp1(fp, function_name, len);\
    else fprint_karat3_3kp2(fp, function_name, len);\
}
#define fprint_karat5(fp, function_name, len) {\
    if(len<5) printf("error (fprint_karat5) len = %d\n",len);\
    else if (len % 5 == 0) fprint_karat5_5k(fp, function_name, len);\
    else printf("error (fprint_karat5) len = %d\n",len);\
}
#define fprint_TC3_256(fp, function_name, len) {\
    if(len%3 == 0 && len >= 6) fprint_TC3_256_3k(fp, function_name, len);\
    else if (len%3 == 1 && len >= 13) fprint_TC3_256_3kp1(fp, function_name, len);\
    else if (len%3 == 2 && len >= 8) fprint_TC3_256_3kp2(fp, function_name, len);\
    else printf("error (fprint_TC3_256) len = %d\n",len);\
}
#define fprint_TC3_128(fp, function_name, len) {\
    if(len%3 == 0 && len >= 3) fprint_TC3_128_3k(fp, function_name, len);\
    else if (len%3 == 1 && len >= 4) fprint_TC3_128_3kp1(fp, function_name, len);\
    else if (len%3 == 2 && len >= 8) fprint_TC3_128_3kp2(fp, function_name, len);\
    else printf("error (fprint_TC3_128) len = %d\n",len);\
}
#define fprint_TC3_64(fp, function_name, len) {\
    if(len%3 == 0 && len >= 3) fprint_TC3_64_3k(fp, function_name, len);\
    else if (len%3 == 1 && len >= 4) fprint_TC3_64_3kp1(fp, function_name, len);\
    else if (len%3 == 2 && len >= 2) fprint_TC3_64_3kp2(fp, function_name, len);\
    else printf("error (fprint_TC3_64) len = %d\n",len);\
}

void fprint_divide_by_x_plus_one_256(FILE * fp);
void fprint_divide_by_x_plus_one_128(FILE * fp);
void fprint_divide_by_x_plus_one_64(FILE * fp);

void fprint_karat_mult_1_PCLMULQDQ(FILE * fp, char * func_name);
void fprint_karat_karat_1_VPCLMULQDQ(FILE * fp, char * func_name);
void fprint_karat_SB_1_VPCLMULQDQ(FILE * fp, char * func_name);
void fprint_SB_karat_1_VPCLMULQDQ(FILE * fp, char * func_name);
void fprint_SB_SB_1_VPCLMULQDQ(FILE * fp, char * func_name);

void fprint_karat2_2k(FILE * fp, char * func_name, uint64_t len);
void fprint_karat2_2kp1(FILE * fp, char * func_name, uint64_t len);

void fprint_karat3_3k(FILE * fp, char * func_name, uint64_t len);
void fprint_karat3_3kp1(FILE * fp, char * func_name, uint64_t len);
void fprint_karat3_3kp2(FILE * fp, char * func_name, uint64_t len);

void fprint_karat5_5k(FILE * fp, char * func_name, uint64_t len);

void fprint_TC3_256_3k(FILE * fp, char * func_name, uint64_t len);
void fprint_TC3_256_3kp1(FILE * fp, char * func_name, uint64_t len);
void fprint_TC3_256_3kp2(FILE * fp, char * func_name, uint64_t len);

void fprint_TC3_128_3k(FILE * fp, char * func_name, uint64_t len);
void fprint_TC3_128_3kp1(FILE * fp, char * func_name, uint64_t len);
void fprint_TC3_128_3kp2(FILE * fp, char * func_name, uint64_t len);

void fprint_TC3_64_3k(FILE * fp, char * func_name, uint64_t len);
void fprint_TC3_64_3kp1(FILE * fp, char * func_name, uint64_t len);
void fprint_TC3_64_3kp2(FILE * fp, char * func_name, uint64_t len);


void gfopt_codegen(uint32_t len);

void best_alg_set(uint32_t dimension, uint8_t alg_num, uint8_t env_num);

#endif
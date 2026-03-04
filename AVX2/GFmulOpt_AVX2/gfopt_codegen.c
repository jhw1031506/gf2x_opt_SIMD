#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gfopt.h"

#ifdef ENV_AVX2
    #ifdef ENV_VPCLMUL
        #include "best_alg/best_alg_result_AVX2_VPCLMUL.h"
        uint8_t best_alg[MAX_LEN+1] = BEST_ALG_AVX2_VPCLMUL;
        char env_name[20] = "VPCLMUL";
    #else
    
        #include "best_alg/best_alg_result_AVX2_PCLMUL.h"
        uint8_t best_alg[MAX_LEN+1] = BEST_ALG_AVX2_PCLMUL;
        char env_name[20] = "PCLMUL";
    #endif
#else
    #ifdef ENV_NEON
    
        #include "best_alg/best_alg_result_NEON.h"
        uint8_t best_alg[MAX_LEN+1] = BEST_ALG_NEON;
        char env_name[20] = "NEON";
    #else
        #include "best_alg/best_alg_result_new.h"
        uint8_t best_alg[MAX_LEN+1] = BEST_ALG_new;
        char env_name[20] = "new";
    #endif
#endif



int compare(const void *a , const void *b) 
{ 
    if( *(int*)a > *(int*)b )       return -1;
    else if( *(int*)a < *(int*)b )  return 1;
    else                            return 0;
} 

uint8_t stack_search(uint32_t * stk, uint32_t len, uint32_t a){
    for(uint32_t i=0;i<len;i++){
        if(stk[i] == a) return 1;
    }
    return 0;
}
#ifdef ENV_AVX2
void stack_push_inner_dim(uint32_t * stk, uint32_t * stk_len, uint32_t p_index, uint8_t * flag_TC){
    uint32_t parents_dim = stk[p_index];
    uint32_t child_dim[3];
    uint32_t child_cnt;
    if     (best_alg[parents_dim] == 1){ //karat2
        uint32_t k = parents_dim/2;
        if(parents_dim % 2 == 0){
            child_cnt = 1;
            child_dim[0] = k;
        }
        else{
            child_cnt = 2;
            child_dim[0] = k;
            child_dim[1] = k+1;
        }
    }
    else if(best_alg[parents_dim] == 2){ //karat3
        uint32_t k = parents_dim/3;
        if(parents_dim % 3 == 0){
            child_cnt = 1;
            child_dim[0] = k;
        }
        else if (parents_dim % 3 == 1){
            child_cnt = 2;
            child_dim[0] = k;
            child_dim[1] = k+1;
        }
        else{
            child_cnt = 2;
            child_dim[0] = k;
            child_dim[1] = k+1;
        }
    }
    else if(best_alg[parents_dim] == 3){ //karat5
        uint32_t k = parents_dim/5;
        if(parents_dim % 5 == 0){
            child_cnt = 1;
            child_dim[0] = k;
        }
        else {
            printf("err. best_alg[%d] = 3 (karat5).\n", parents_dim);
            return;
        }
    }
    else if(best_alg[parents_dim] == 4){ //TC3_256
        flag_TC[0] = 1;
        uint32_t k = parents_dim/3;
        if(parents_dim % 3 == 0){
            child_cnt = 2;
            child_dim[0] = k;
            child_dim[1] = k+2;
        }
        else if (parents_dim % 3 == 1){
            child_cnt = 3;
            child_dim[0] = k-1;
            child_dim[1] = k+1;
            child_dim[2] = k+2;
        }
        else{
            child_cnt = 3;
            child_dim[0] = k;
            child_dim[1] = k+1;
            child_dim[2] = k+2;
        }
    }
    else if(best_alg[parents_dim] == 5){ //TC3_128
        flag_TC[1] = 1;
        uint32_t k = parents_dim/3;
        if(parents_dim % 3 == 0){
            child_cnt = 2;
            child_dim[0] = k;
            child_dim[1] = k+1;
        }
        else if (parents_dim % 3 == 1){
            child_cnt = 2;
            child_dim[0] = k;
            child_dim[1] = k+1;
        }
        else{
            child_cnt = 3;
            child_dim[0] = k;
            child_dim[1] = k+1;
            child_dim[2] = k+2;
        }
    }
    else if(best_alg[parents_dim] == 6){ //TC3_64
        flag_TC[2] = 1;
        uint32_t k = parents_dim/3;
        if(parents_dim % 3 == 0){
            child_cnt = 2;
            child_dim[0] = k;
            child_dim[1] = k+1;
        }
        else if (parents_dim % 3 == 1){
            child_cnt = 2;
            child_dim[0] = k;
            child_dim[1] = k+1;
        }
        else{
            child_cnt = 1;
            child_dim[0] = k+1;
        }
    }
    else{
        printf("err. best_alg[%d] = %d.\n", parents_dim, best_alg[parents_dim]);
        printf("parent_index = %d\n",p_index);
        return;
    }

    for(uint32_t i=0;i<child_cnt;i++){
        if((child_dim[i] != 1) && (!stack_search(stk, *stk_len, child_dim[i]))){
            stk[*stk_len] = child_dim[i];
            (*stk_len) ++;
        }
    }
}
#endif
void gfopt_codegen(uint32_t len){
    //flag_TC[0]: TC256
    //flag_TC[1]: TC128
    //flag_TC[2]: TC64
    uint8_t flag_TC[3]={0};
    printf("generating GF(2)[x] multiplication code for len = %d..\n", len);
    if(len > MAX_LEN){
        printf("gfopt_codegen error : len = %d > MAX_LEN = %d\n", len, MAX_LEN);
        return;
    }
    if(len == 0){
        printf("gfopt_codegen error : len = 0");
        return;
    }
    char func_name[20];
    char target_name[100];
     
    sprintf(target_name, "result/gfopt_%s_%d.c",env_name,len);
    FILE* fp = fopen(target_name, "w");
    FILE* fpr1 = fopen("result/result_head", "rb");
    char file;
    #if ENV_AVX2
    fprintf(fp,"#include <immintrin.h>\n");
    #endif
    fprintf(fp,"#define GFLEN %d\n",len);
    fprintf(fp,"#define GFMUL(C, A, B) gfmul_%d(C, A, B)\n",len);
	while (!feof(fpr1))
	{
		if(fread(&file, 1, 1, fpr1)!=1) break;
		fwrite(&file, 1, 1, fp);
	}
    fclose(fpr1);

    uint32_t stack[MAX_LEN];
    uint32_t stack_len = 1, parents_index = 0;

    //set stack
    stack[0] = len;
    while(parents_index < stack_len){
        stack_push_inner_dim(stack, &stack_len, parents_index, flag_TC);
        parents_index ++;
        qsort(stack, stack_len, sizeof(stack[0]), compare);
    }
//divide_by_x_plus_one (internal function of Toom-Cook)
    #ifdef ENV_AVX2
    if(flag_TC[0]){
        fprint_divide_by_x_plus_one_256(fp);
    }
    if(flag_TC[1]){
        fprint_divide_by_x_plus_one_128(fp);
    }
    if(flag_TC[2]){
        fprint_divide_by_x_plus_one_64(fp);
    }
    #endif
//basemul (len = 1)
    #ifdef ENV_AVX2
    sprintf(func_name,"gfmul_1");
    #ifdef ENV_VPCLMUL
    switch(best_alg[1]){
    case 0:
        fprintf(fp, "//len = 1: karat_karat_PCLMULQDQ\n");
        fprint_karat_mult_1_PCLMULQDQ(fp, func_name);
        break;
    case 1:
        fprintf(fp, "//len = 1: karat_karat_256\n");
        fprint_karat_karat_1_VPCLMULQDQ(fp, func_name);
        break;
    case 2:
        fprintf(fp, "//len = 1: karat_SB_256\n");
        fprint_karat_SB_1_VPCLMULQDQ(fp, func_name);
        break;
    case 3:
        fprintf(fp, "//len = 1: SB_karat_256\n");
        fprint_SB_karat_1_VPCLMULQDQ(fp, func_name);
        break;
    case 4:
        fprintf(fp, "//len = 1: SB_SB_256\n");
        fprint_SB_SB_1_VPCLMULQDQ(fp, func_name);
        break;
    default:
        printf("err. best_alg[1] = %d\n", best_alg[1]);
    }
    #elif
    
    #endif
#endif


//polymul (len = 2, 3, ...)
    #ifdef ENV_AVX2
    for(int i = (int)stack_len-1; i>=0; i--){
        sprintf(func_name,"gfmul_%d",stack[i]);
        switch(best_alg[stack[i]]){
        case 1:
            fprintf(fp, "//len = %d: 2-Karatsuba\n",stack[i]);
            fprint_karat2(fp, func_name, stack[i]);
            break;
        case 2:
            fprintf(fp, "//len = %d: 3-Karatsuba\n",stack[i]);
            fprint_karat3(fp, func_name, stack[i]);
            break;
        case 3:
            fprintf(fp, "//len = %d: 5-Karatsuba\n",stack[i]);
            fprint_karat5(fp, func_name, stack[i]);
            break;
        case 4:
            fprintf(fp, "//len = %d: TC3_256\n",stack[i]);
            fprint_TC3_256(fp, func_name, stack[i]);
            break;
        case 5:
            fprintf(fp, "//len = %d: TC3_128\n",stack[i]);
            fprint_TC3_128(fp, func_name, stack[i]);
            break;
        case 6:
            fprintf(fp, "//len = %d: TC3_64\n",stack[i]);
            fprint_TC3_64(fp, func_name, stack[i]);
            break;
        default:
            printf("err. best_alg[%d] = %d\n", stack[i], best_alg[stack[i]]);
        }
    }    
    FILE* fpr2 = fopen("result/result_tail", "rb");
	while (!feof(fpr2))
	{
		if(fread(&file, 1, 1, fpr2)!=1) break;
		fwrite(&file, 1, 1, fp);
	}
    fclose(fpr2);
    fclose(fp);
    #endif
}
int main(int argc, char *argv[]){
    char strsys1[200], strsys2[200];
    if (argc < 2){
        printf("Usage: %s <len>\n", argv[0]);
        return 1;
    }
    
    char str_options[200]=" -mpclmul -msse2 -mavx2 ";
    #ifdef ENV_VPCLMUL
    strcat(str_options, "-mvpclmulqdq ");
    #endif

    for (int i = 1; i < argc; i++){
        int len = atoi(argv[i]);
        gfopt_codegen(len);
        sprintf(strsys1,"gcc result/gfopt_%s_%d.c -o result/gfopt_%s_%d -O3 -std=c99 -funroll-all-loops  -pedantic -Wall -Wextra",env_name,len,env_name,len);
        strcat(strsys1, str_options);
        if(system(strsys1)) printf("gfopt_%s_%d.c compile err\n",env_name,len);
        sprintf(strsys2,"./result/gfopt_%s_%d",env_name,len);
        if(system(strsys2)) printf("result/gfopt_%s_%d err\n",env_name,len);
    }
    printf("Code generation complete. Output saved to  result/gfopt_%s_*.c\n", env_name);

    return 0;
}
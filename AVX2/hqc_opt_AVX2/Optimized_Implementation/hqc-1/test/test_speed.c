#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/api.h"
#include "../src/parameters.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "../src/vector.h"



#define NTESTS 100000

uint64_t t[NTESTS];
uint64_t t_vecmul;
#if PARAM_SECURITY == 128
 #define PARAM_LV 1
#elif PARAM_SECURITY == 192
 #define PARAM_LV 3
#elif PARAM_SECURITY == 256
 #define PARAM_LV 5
#endif
int main(void)
{
	printf("\n");
	printf("*********************\n");
	printf("**** HQC-%d test****\n", PARAM_LV);
	printf("*********************\n");

	printf("\n");
	printf("N: %d   ", PARAM_N);
	printf("N1: %d   ", PARAM_N1);
	printf("N2: %d   ", PARAM_N2);
	printf("OMEGA: %d   ", PARAM_OMEGA);
	printf("OMEGA_R: %d   ", PARAM_OMEGA_R);
	printf("Failure rate: 2^-%d   ", PARAM_DFR_EXP);
	printf("Sec: %d bits", PARAM_SECURITY);
	printf("\n\n");
  #if GFMUL_VER == 1
  printf("ORIGINAL MUL: PCLMUL\n\n");
  #elif GFMUL_VER == 2
  printf("ORIGINAL MUL: VPCLMUL\n\n");
  #elif GFMUL_VER == 3
  printf("GFmulOpt MUL: PCLMUL\n\n");
  #elif GFMUL_VER == 4
  printf("GFmulOpt MUL: VPCLMUL\n\n");
  #elif GFMUL_VER == 5
  printf("gf2x library: PCLMUL\n\n");
  #endif

	unsigned char pk[PUBLIC_KEY_BYTES];
	unsigned char sk[SECRET_KEY_BYTES];
	unsigned char ct[CIPHERTEXT_BYTES];
	unsigned char key1[SHARED_SECRET_BYTES];
	unsigned char key2[SHARED_SECRET_BYTES];

    int i;
    
    t_vecmul=0;
    printf("test...");
    for (i = 0; i < NTESTS/1000; ++i){
        crypto_kem_keypair(pk, sk);
        crypto_kem_enc(ct, key1, pk);
        crypto_kem_dec(key2, ct, sk);

        for(int i = 0 ; i < SHARED_SECRET_BYTES ; ++i){
            if(key1[i]!=key2[i]){
                printf("\n\nsecret1: ");
                for(int j = 0 ; j < SHARED_SECRET_BYTES ; ++j){
                    printf("%x", key1[j]);
                }

                printf("\nsecret2: ");
                for(int j = 0 ; j < SHARED_SECRET_BYTES ; ++j){
                    printf("%x", key2[j]);
                }

                printf("\n\ntest failed\n");
                exit(0);
                break;
            }
        }

    }
    printf(" ok\n\n");

    for (i = 0; i < NTESTS; ++i)
    {
        t[i] = cpucycles();
        crypto_kem_keypair(pk, sk);
    }
    print_results("Keypair", t, NTESTS);
    printf(" └─ polynomial multiplication\n    average: %ld cycles/ticks\n\n",t_vecmul/NTESTS); 
    

    t_vecmul=0;
    for (i = 0; i < NTESTS; ++i)
    {
        t[i] = cpucycles();
        crypto_kem_enc(ct, key1, pk);
    }
    print_results("Encapsulation", t, NTESTS);
    printf(" └─ polynomial multiplication\n    average: %ld cycles/ticks\n\n",t_vecmul/NTESTS); 
    
    t_vecmul=0;
    for (i = 0; i < NTESTS; ++i)
    {
        t[i] = cpucycles();
        crypto_kem_dec(key2, ct, sk);
    }
    print_results("Decapsulation", t, NTESTS);
    printf(" └─ polynomial multiplication\n    average: %ld cycles/ticks\n",t_vecmul/NTESTS); 
    
    return 0;
}

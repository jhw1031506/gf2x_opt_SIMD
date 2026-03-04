#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/api.h"
#include "../src/parameters.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "../src/vector.h"

#include <unistd.h>
#include <sched.h>
#define NTESTS 100000

uint64_t t[NTESTS];
int64_t t_vecmul;

#if PARAM_SECURITY == 128
 #define PARAM_LV 1
#elif PARAM_SECURITY == 192
 #define PARAM_LV 3
#elif PARAM_SECURITY == 256
 #define PARAM_LV 5
#endif

int main(void)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);      
    CPU_SET(0, &cpuset);    

    pid_t pid = getpid();
    if (sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset) == -1) {
        perror("sched_setaffinity");
        exit(EXIT_FAILURE);
    }
    printf("\n");
    printf("*********************\n");
    printf("** HQC-%d test **\n", PARAM_LV);
    printf("*********************\n");

    printf("\n");
    printf("N: %d   ", PARAM_N);
    printf("N1: %d   ", PARAM_N1);
    printf("N2: %d   ", PARAM_N2);
    printf("OMEGA: %d   ", PARAM_OMEGA);
    printf("OMEGA_R: %d   ", PARAM_OMEGA_R);
    printf("Failure rate: 2^-%d   ", PARAM_DFR_EXP);
    printf("Sec: %d bits", PARAM_SECURITY);
    printf("\n");
    unsigned char pk[PUBLIC_KEY_BYTES];
    unsigned char sk[SECRET_KEY_BYTES];
    unsigned char ct[CIPHERTEXT_BYTES];
    unsigned char key1[SHARED_SECRET_BYTES];
    unsigned char key2[SHARED_SECRET_BYTES];
    printf("\nNTESTS=%d\n\n",NTESTS);

    int i;
    uint64_t start;

    uint64_t t_keypair = 0, t_enc = 0, t_dec = 0;

    uint64_t t_vecmul_keypair = 0, t_vecmul_enc = 0, t_vecmul_dec = 0;

    uint64_t t_overhead = cpucycles_overhead();
    printf("test...");
    for (i = 0; i < NTESTS/100; ++i){
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

    
        t_vecmul=0;

        start = cpucycles();
        crypto_kem_keypair(pk, sk);
        t_keypair += cpucycles() - start - t_overhead;
        t_vecmul_keypair += t_vecmul;

        t_vecmul = 0;

        start = cpucycles();
        crypto_kem_enc(ct, key1, pk);
        t_enc += cpucycles() - start - t_overhead;
        t_vecmul_enc += t_vecmul;

        t_vecmul = 0;

        start = cpucycles();
        crypto_kem_dec(key2, ct, sk);
        t_dec += cpucycles() - start - t_overhead;
        t_vecmul_dec += t_vecmul;
    }
    printf("Keypair: %ld cycles/ticks\n",t_keypair/(uint64_t)NTESTS);
    printf(" └─ polynomial multiplication\n    average: %ld cycles/ticks\n",t_vecmul_keypair/NTESTS); 
    
    printf("Encapsulation: %ld cycles/ticks\n",t_enc/NTESTS);
    printf(" └─ polynomial multiplication\n    average: %ld cycles/ticks\n",t_vecmul_enc/NTESTS); 

    printf("Decapsulation: %ld cycles/ticks\n",t_dec/NTESTS);
    printf(" └─ polynomial multiplication\n    average: %ld cycles/ticks\n",t_vecmul_dec/NTESTS); 

    return 0;
}

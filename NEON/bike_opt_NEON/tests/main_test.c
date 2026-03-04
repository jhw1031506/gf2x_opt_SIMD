/* Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0"
 *
 * Written by Nir Drucker, Shay Gueron and Dusan Kostic,
 * AWS Cryptographic Algorithms Group.
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include <unistd.h>
#include <sched.h>

#include "gf2x.h"
#include "kem.h"
#include "measurements.h"
#include "utilities.h"
#include "cpu_features.h"

#include "cpucycles.h"

#if !defined(NUM_OF_TESTS)
#  define NUM_OF_TESTS 1000
#endif

typedef struct magic_number_s {
  uint64_t val[4];
} magic_number_t;

#define STRUCT_WITH_MAGIC(name, size) \
  struct {                            \
    magic_number_t magic1;            \
    uint8_t        val[size];         \
    magic_number_t magic2;            \
  }(name) = {magic, {0}, magic};

#define CHECK_MAGIC(param)                                          \
  if((0 != memcmp((param).magic1.val, magic.val, sizeof(magic))) || \
     (0 != memcmp((param).magic2.val, magic.val, sizeof(magic)))) { \
    printf("Magic is incorrect for param\n");                       \
  }
  
uint64_t cpucycles_overhead(void) {
    uint64_t t0, t1, overhead = 0;
    unsigned int i;

    for(i=0;i<100000;i++) {
        t0 = cpucycles();
        __asm__ volatile("");
        t1 = cpucycles();
        overhead += t1 - t0;
    }
    return overhead/100000;
}

////////////////////////////////////////////////////////////////
//                 Main function for testing
////////////////////////////////////////////////////////////////
  uint64_t t_keypair_mul=0, t_enc_mul=0, t_dec_mul=0;/////////////
  uint64_t t_over;/////////////
int main(void)
{
  // Initialize the CPU features flags
  cpu_features_init();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);      // CPU 집합 초기화
    CPU_SET(0, &cpuset);    // CPU 0번을 집합에 추가
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("sched_setaffinity");
        exit(EXIT_FAILURE);
    }
    printf("Process is bound to CPU: ");
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset))
            printf("%d ", i);
    }
    printf("\n");
  printf("LEVEL = %d\n\n",LEVEL);
  #if GFMUL_VER == 0
  printf("ORIGINAL MUL: non-NEON\n");
  #elif GFMUL_VER == 1
  printf("ORIGINAL MUL: NEON\n");
  #elif GFMUL_VER == 2
  printf("GFmulOpt: NEON\n");
  #elif GFMUL_VER == 3
  printf("gf2x library:non-NEON\n");
  #endif
  
  volatile uint64_t start, fin, t_keypair=0, t_encap=0, t_decap=0;/////////////
    
#if defined(FIXED_SEED)
  srand(0);
#else
  srand(time(NULL));
#endif

  magic_number_t magic = {0xa1234567b1234567, 0xc1234567d1234567,
                          0xe1234567f1234567, 0x0123456711234567};

  STRUCT_WITH_MAGIC(sk, sizeof(sk_t));
  STRUCT_WITH_MAGIC(pk, sizeof(pk_t));
  STRUCT_WITH_MAGIC(ct, sizeof(ct_t));
  STRUCT_WITH_MAGIC(k_enc, sizeof(ss_t)); // shared secret after decapsulate
  STRUCT_WITH_MAGIC(k_dec, sizeof(ss_t)); // shared secret after encapsulate

  for(size_t i = 1; i <= NUM_OF_TESTS; ++i) {
    int res = 0;
    //printf("Code test: %lu\n", i);
    // Key generation
    start = cpucycles();/////////////
    //MEASURE("  keypair", res = crypto_kem_keypair(pk.val, sk.val););
    res = crypto_kem_keypair(pk.val, sk.val);
    fin = cpucycles();/////////////
    t_keypair += fin - start - t_over;/////////////
    if(res != 0) {
      printf("Keypair failed with error: %d\n", res);
      continue;
    }

    uint32_t dec_rc = 0;

    // Encapsulate
    start = cpucycles();/////////////
    // MEASURE("  encaps", res = crypto_kem_enc(ct.val, k_enc.val, pk.val););
    res = crypto_kem_enc(ct.val, k_enc.val, pk.val);
    fin = cpucycles();/////////////
    t_encap += fin - start - t_over;/////////////
    if(res != 0) {
      printf("encapsulate failed with error: %d\n", res);
      continue;
    }

    // Decapsulate
    start = cpucycles();/////////////
    //MEASURE("  decaps", dec_rc = crypto_kem_dec(k_dec.val, ct.val, sk.val););
    crypto_kem_dec(k_dec.val, ct.val, sk.val);
    fin = cpucycles();/////////////
    t_decap += fin - start - t_over;/////////////

    // Check test status
    if(dec_rc != 0) {
      printf("Decoding failed after %ld code tests!\n", i);
    } else {
      if(secure_cmp(k_enc.val, k_dec.val, sizeof(k_dec.val) / sizeof(uint64_t))) {
        // printf("Success! decapsulated key is the same as encapsulated "
        //        "key!\n");
      } else {
        printf("Failure! decapsulated key is NOT the same as encapsulated "
               "key!\n");
      }
    }

    // Check magic numbers (memory overflow) 
    CHECK_MAGIC(sk);
    CHECK_MAGIC(pk);
    CHECK_MAGIC(ct);
    CHECK_MAGIC(k_enc);
    CHECK_MAGIC(k_dec);

    print("Initiator's generated key (K) of 256 bits = ", (uint64_t *)k_enc.val,
          SIZEOF_BITS(k_enc.val));
    print("Responder's computed key (K) of 256 bits  = ", (uint64_t *)k_dec.val,
          SIZEOF_BITS(k_enc.val));
  }
  printf("keypair: %ld\n", t_keypair/NUM_OF_TESTS);
  printf("\tgfmul  : %ld\n", t_keypair_mul/NUM_OF_TESTS);

  printf("encaps : %ld\n", t_encap/NUM_OF_TESTS);
  printf("\tgfmul  : %ld\n", t_enc_mul/NUM_OF_TESTS);

  printf("decaps : %ld\n", t_decap/NUM_OF_TESTS);
  printf("\tgfmul  : %ld\n", t_dec_mul/NUM_OF_TESTS);
  

  return 0;
}

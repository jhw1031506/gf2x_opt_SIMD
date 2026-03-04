/* Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0"
 *
 * Written by Nir Drucker, Shay Gueron and Dusan Kostic,
 * AWS Cryptographic Algorithms Group.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gf2x.h"
#include "cpucycles.h"
#include "kem.h"
#include "measurements.h"
#include "utilities.h"
#include "cpu_features.h"

#if !defined(NUM_OF_TESTS)
#  define NUM_OF_TESTS 100000
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
  uint64_t t0, t1, overhead = -1LL;
  unsigned int i;

  for(i=0;i<100000;i++) {
    t0 = cpucycles();
    __asm__ volatile("");
    t1 = cpucycles();
    if(t1 - t0 < overhead)
      overhead = t1 - t0;
  }

  return overhead;
}

////////////////////////////////////////////////////////////////
//                 Main function for testing
////////////////////////////////////////////////////////////////
  uint64_t t_keypair_mul=0, t_enc_mul=0, t_dec_mul=0;/////////////
  uint64_t t_over;/////////////
int main(void)
{
  printf("LEVEL = %d\n\n",LEVEL);
  #if GFMUL_VER == 0
  printf("ORIGINAL MUL\n");
  #elif GFMUL_VER == 1
  printf("GFmulOpt MUL\n");
  #elif GFMUL_VER == 2
  printf("CHES2021 MUL\n");
  #elif GFMUL_VER == 3
  printf("gf2x library\n");
  #endif
  #if VPCLMUL == 0
  printf("PCLMUL\n");
  #elif VPCLMUL == 1
  printf("VPCLMUL\n");
  #endif
  // Initialize the CPU features flags
  cpu_features_init();
  volatile uint64_t start, fin, t_keypair=0, t_encap=0, t_decap=0;/////////////

#if defined(FIXED_SEED)
  srand(0);
#else
  srand(time(NULL));
#endif

  t_over=cpucycles_overhead();
  magic_number_t magic = {0xa1234567b1234567, 0xc1234567d1234567,
                          0xe1234567f1234567, 0x0123456711234567};

  STRUCT_WITH_MAGIC(sk, sizeof(sk_t));
  STRUCT_WITH_MAGIC(pk, sizeof(pk_t));
  STRUCT_WITH_MAGIC(ct, sizeof(ct_t));
  STRUCT_WITH_MAGIC(k_enc, sizeof(ss_t)); // shared secret after decapsulate
  STRUCT_WITH_MAGIC(k_dec, sizeof(ss_t)); // shared secret after encapsulate

  for(size_t i = 1; i <= NUM_OF_TESTS; ++i) {
    int res = 0;

    // printf("Code test: %lu\n", i);

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
  printf(" \tgfmul  : %ld\n", t_dec_mul/NUM_OF_TESTS);
  

  return 0;
}

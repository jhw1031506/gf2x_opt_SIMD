/*
 * Written by Ming-Shing Chen and Tung Chou
 */

#ifndef _RKARA3_MUL_H_
#define _RKARA3_MUL_H_

#if defined(CRYPTO_NAMESPACE)
#define rkara3_mul_1536  CRYPTO_NAMESPACE(rkara3_mul_1536)
#define rkara3_mul_12288  CRYPTO_NAMESPACE(rkara3_mul_12288)
#define rkara3_mul_12352  CRYPTO_NAMESPACE(rkara3_mul_12352)
#define rkara3_mul_24704  CRYPTO_NAMESPACE(rkara3_mul_24704)
#endif


#include "stdint.h"
#include "stdlib.h"


void rkara3_mul_1536(uint64_t *c, const uint64_t *a, const uint64_t *b);
void rkara3_mul_12288(uint64_t *c, const uint64_t *a, const uint64_t *b);
void rkara3_mul_12352(uint64_t *c, const uint64_t *a, const uint64_t *b);
void rkara3_mul_24704(uint64_t *c, const uint64_t *a, const uint64_t *b);
void rkara3_mul_12352_my(uint64_t *c, const uint64_t *a, const uint64_t *b);
void rkara3_mul_24704_my(uint64_t *c, const uint64_t *a, const uint64_t *b);
void rkara3_mul_40976_my(uint64_t *c, const uint64_t *a, const uint64_t *b);
void rkara3_mul_12352_my_vpclmul(uint64_t *c, const uint64_t *a, const uint64_t *b);
void rkara3_mul_24704_my_vpclmul(uint64_t *c, const uint64_t *a, const uint64_t *b);
void rkara3_mul_40976_my_vpclmul(uint64_t *c, const uint64_t *a, const uint64_t *b);
void karatzuba_ches(uint64_t *c,
                         const uint64_t *a,
                        const uint64_t *b,
                         const size_t    qwords_len,
                        const size_t    qwords_len_pad,
                        uint64_t *         sec_buf);
#endif

void gf2x_mul_4096(uint64_t *c, const uint64_t *a, const uint64_t *b);
void gf2x_mul_8192(uint64_t *c, const uint64_t *a, const uint64_t *b);
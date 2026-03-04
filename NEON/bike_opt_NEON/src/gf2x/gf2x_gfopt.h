#ifndef _GF2X_OPT_H_
#define _GF2X_OPT_H_



#include "stdint.h"
#include "stdlib.h"
#include <arm_neon.h>

void gfmul_97(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128);
void gfmul_193(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128);
void gfmul_321(poly8x16_t *Out,  const  poly8x16_t *A128,  const  poly8x16_t *B128);

#endif
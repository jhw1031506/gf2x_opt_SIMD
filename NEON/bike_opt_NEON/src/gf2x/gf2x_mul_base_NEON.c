#include "gf2x_internal.h"
#include "utilities.h"
#include "bike_defs.h"
#include <arm_neon.h>


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

//len = 1: karat_NEON
void gf2x_mul_base_NEON(uint64_t * Out, const uint64_t * a, const uint64_t * b){
	poly8x8_t al, ah, bl, bh;
	poly8x8_t alpah, blpbh;
	poly8x16_t D0={0}, D1={0}, D2={0};
	ah = vget_high_p8(*(const poly8x16_t*)a);
	al = vget_low_p8(*(const poly8x16_t*)a);
	bh = vget_high_p8(*(const poly8x16_t*)b);
	bl = vget_low_p8(*(const poly8x16_t*)b);
	alpah=vadd_p8(al,ah);
	blpbh=vadd_p8(bl,bh);
    // schmul64_NEON(&D1, &alpah, &blpbh);
    // schmul64_NEON(&D0, &al, &bl);
    // schmul64_NEON(&D2, &ah, &bh);
    D1 = (poly8x16_t)vmull64_a72(*(uint64_t *)&alpah, *(uint64_t *)&blpbh);
    D0 = (poly8x16_t)vmull64_a72(*(uint64_t *)&al, *(uint64_t *)&bl);
    D2 = (poly8x16_t)vmull64_a72(*(uint64_t *)&ah, *(uint64_t *)&bh);

    D1=vaddq_p8(D1,D0);
    D1=vaddq_p8(D1,D2);

    ((poly8x16_t*)Out)[0]=vcombine_p8(vget_low_p8(D0),vadd_p8(vget_high_p8(D0),vget_low_p8(D1)));
    ((poly8x16_t*)Out)[1]=vcombine_p8(vadd_p8(vget_low_p8(D2),vget_high_p8(D1)),vget_high_p8(D2));
}

// c = a^2
void gf2x_sqr_NEON(OUT dbl_pad_r_t *c, IN const pad_r_t *a)
{
  const uint64_t *a64 = (const uint64_t *)a;
  uint64_t *      c64 = (uint64_t *)c;

  for(size_t i = 0; i < R_QWORDS; i+=2) {
    gf2x_mul_base_NEON(&c64[2 * i], &a64[i], &a64[i]);
  }
}

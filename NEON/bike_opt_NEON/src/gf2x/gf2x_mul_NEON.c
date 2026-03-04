#include <arm_neon.h>
#include <assert.h>
#include <stdint.h>

// 기존 헤더 파일들 (프로젝트 환경에 맞게 유지)
#include "cleanup.h"
#include "gf2x_internal.h"

// NEON 정의
#define REG_T_NEON       uint64x2_t
#define REG_QWORDS_NEON  2  // 128-bit NEON 레지스터는 64-bit 정수 2개를 처리함

void karatzuba_add1_NEON(OUT uint64_t *alah,
                         OUT uint64_t *blbh,
                         IN const uint64_t *a,
                         IN const uint64_t *b,
                         IN const size_t    qwords_len)
{
    assert(qwords_len % REG_QWORDS_NEON == 0);

    REG_T_NEON va0, va1, vb0, vb1;

    for(size_t i = 0; i < qwords_len; i += REG_QWORDS_NEON) {
        // 데이터 로드 (Load)
        va0 = vld1q_u64(&a[i]);
        va1 = vld1q_u64(&a[i + qwords_len]);
        vb0 = vld1q_u64(&b[i]);
        vb1 = vld1q_u64(&b[i + qwords_len]);

        // XOR 연산 후 저장 (Store)
        vst1q_u64(&alah[i], veorq_u64(va0, va1));
        vst1q_u64(&blbh[i], veorq_u64(vb0, vb1));
    }
}

void karatzuba_add2_NEON(OUT uint64_t *z,
                         IN const uint64_t *x,
                         IN const uint64_t *y,
                         IN const size_t    qwords_len)
{
    assert(qwords_len % REG_QWORDS_NEON == 0);

    REG_T_NEON vx, vy;

    for(size_t i = 0; i < qwords_len; i += REG_QWORDS_NEON) {
        vx = vld1q_u64(&x[i]);
        vy = vld1q_u64(&y[i]);

        vst1q_u64(&z[i], veorq_u64(vx, vy));
    }
}

void karatzuba_add3_NEON(OUT uint64_t *c,
                         IN const uint64_t *mid,
                         IN const size_t    qwords_len)
{
    assert(qwords_len % REG_QWORDS_NEON == 0);

    REG_T_NEON vr0, vr1, vr2, vr3, vt;

    uint64_t *c0 = c;
    uint64_t *c1 = &c[qwords_len];
    uint64_t *c2 = &c[2 * qwords_len];
    uint64_t *c3 = &c[3 * qwords_len];

    for(size_t i = 0; i < qwords_len; i += REG_QWORDS_NEON) {
        vr0 = vld1q_u64(&c0[i]);
        vr1 = vld1q_u64(&c1[i]);
        vr2 = vld1q_u64(&c2[i]);
        vr3 = vld1q_u64(&c3[i]);
        vt  = vld1q_u64(&mid[i]);

        // c1[i] = vt ^ vr0 ^ vr1
        vst1q_u64(&c1[i], veorq_u64(vt, veorq_u64(vr0, vr1)));
        
        // c2[i] = vt ^ vr2 ^ vr3
        vst1q_u64(&c2[i], veorq_u64(vt, veorq_u64(vr2, vr3)));
    }
}

// c = a mod (x^r - 1)
void gf2x_red_NEON(OUT pad_r_t *c, IN const dbl_pad_r_t *a)
{
    const uint64_t *a64 = (const uint64_t *)a;
    uint64_t * c64 = (uint64_t *)c;
    size_t i;
    for(i = 0; i < R_QWORDS - 1; i += REG_QWORDS_NEON) {
        REG_T_NEON vt0 = vld1q_u64(&a64[i]);
        REG_T_NEON vt1 = vld1q_u64(&a64[i + R_QWORDS]);
        REG_T_NEON vt2 = vld1q_u64(&a64[i + R_QWORDS - 1]);

        vt1 = vshlq_n_u64(vt1, LAST_R_QWORD_TRAIL); 
        vt2 = vshrq_n_u64(vt2, LAST_R_QWORD_LEAD);

        vt0 = veorq_u64(vt0, vorrq_u64(vt1, vt2));

        vst1q_u64(&c64[i], vt0);
    }
    if(i == R_QWORDS - 1){
    uint64_t vt0 = a64[i];
    uint64_t vt1 = a64[i + R_QWORDS];
    uint64_t vt2 = a64[i + R_QWORDS - 1];

    vt1 = (vt1 << LAST_R_QWORD_TRAIL);
    vt2 = (vt2 >> LAST_R_QWORD_LEAD);

    vt0 ^= (vt1 | vt2);

    c64[i] = vt0;
    }

    c64[R_QWORDS - 1] &= LAST_R_QWORD_MASK;

    secure_clean((uint8_t *)&c64[R_QWORDS],
                 (R_PADDED_QWORDS - R_QWORDS) * sizeof(uint64_t));
}
/*
 * Written by Ming-Shing Chen and Tung Chou
 */

#include "stdint.h"
#include "rkara3_mul.h"
#include "gf2x_gfopt.h"
#include "gf2x_mul_base_pclmul.h"

#include <immintrin.h>


#include <string.h>



static inline
__m256i msbyte( __m256i a ) { return _mm256_permute4x64_epi64(_mm256_srli_si256(a,15),0xfe); } // 11,11,11,10

static inline
__m256i shl_1( __m256i a, __m256i a_minus_1 ) { return _mm256_slli_epi16(a,1)|_mm256_srli_epi16(a_minus_1,7); }

static inline
__m256i shl_2( __m256i a, __m256i a_minus_1 ) { return _mm256_slli_epi16(a,2)|_mm256_srli_epi16(a_minus_1,6); }

static
void shl_1_test( uint8_t *b , const uint8_t *a , int len )
{
  __m256i _a0 = _mm256_loadu_si256((const __m256i*)a);
  __m256i a15 = _mm256_permute4x64_epi64(_mm256_srli_si256(_a0,15),0x4f); // 01,00,11,11
  __m256i _a_1 = _mm256_slli_si256(_a0,1)|a15;
  _mm256_storeu_si256((__m256i*)b, shl_1(_a0,_a_1) );

  for(int i=32;i<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    __m256i a_1 = _mm256_loadu_si256((const __m256i*)(a+i-1));
    _mm256_storeu_si256((__m256i*)(b+i), shl_1(a0,a_1) );
  }
}

static
void shl_2_test( uint8_t *b , const uint8_t *a , int len )
{
  __m256i _a0 = _mm256_loadu_si256((const __m256i*)a);
  __m256i a15 = _mm256_permute4x64_epi64(_mm256_srli_si256(_a0,15),0x4f); // 01,00,11,11
  __m256i _a_1 = _mm256_slli_si256(_a0,1)|a15;
  _mm256_storeu_si256((__m256i*)b, shl_2(_a0,_a_1) );

  for(int i=32;i<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    __m256i a_1 = _mm256_loadu_si256((const __m256i*)(a+i-1));
    _mm256_storeu_si256((__m256i*)(b+i), shl_2(a0,a_1) );
  }
}

////////////////////////

static inline
void add( uint8_t *c , const uint8_t *a , const uint8_t *b, int len )
{
  for(int i=0;i<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    __m256i b0 = _mm256_loadu_si256((const __m256i*)(b+i));
    _mm256_storeu_si256((__m256i*)(c+i), a0^b0 );
  }
}

static inline
void cpy( uint8_t *c , const uint8_t *a , int len ) {
  for(int i=0;i<len;i+=32) _mm256_storeu_si256((__m256i*)(c+i), _mm256_loadu_si256((const __m256i*)(a+i)) );
}

//////////////////////////

static inline
__m256i shr_1( __m256i a, __m256i a_plus_1 ) { return _mm256_srli_epi16(a,1)|_mm256_slli_epi16(a_plus_1,7); }

static
void shr_1_test( uint8_t *b , const uint8_t *a , int len )
{
  for(int i=0;i<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    __m256i a_1 = _mm256_loadu_si256((const __m256i*)(a+i+1));
    _mm256_storeu_si256((__m256i*)(b+i), shr_1(a0,a_1) );
  }
}

static
void add_shr1( uint8_t *a , int len )
{
  for(int i=0;i<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    __m256i a_1 = _mm256_loadu_si256((const __m256i*)(a+i+1));
    _mm256_storeu_si256((__m256i*)(a+i), a0^shr_1(a0,a_1) );
  }
}

static inline
__m256i shr_2( __m256i a, __m256i a_plus_1 ) { return _mm256_srli_epi16(a,2)|_mm256_slli_epi16(a_plus_1,6); }

static
void add_shr2( uint8_t *a , int len )
{
  for(int i=0;i<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    __m256i a_1 = _mm256_loadu_si256((const __m256i*)(a+i+1));
    _mm256_storeu_si256((__m256i*)(a+i), a0^shr_2(a0,a_1) );
  }
}

static
void shr_2_test( uint8_t *b , const uint8_t *a , int len )
{
  for(int i=0;i<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    __m256i a_1 = _mm256_loadu_si256((const __m256i*)(a+i+1));
    _mm256_storeu_si256((__m256i*)(b+i), shr_2(a0,a_1) );
  }
}


static inline
__m256i shr_4( __m256i a, __m256i a_plus_1 ) { return _mm256_srli_epi16(a,4)|_mm256_slli_epi16(a_plus_1,4); }

void add_shr4( uint8_t *a , int len )
{
  for(int i=0;i<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    __m256i a_1 = _mm256_loadu_si256((const __m256i*)(a+i+1));
    _mm256_storeu_si256((__m256i*)(a+i), a0^shr_4(a0,a_1) );
  }
}


static inline
void add_shr_jbyte( uint8_t *a , int len , int j)
{
  for(int i=0;i+j<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    _mm256_storeu_si256((__m256i*)(a+i), a0^_mm256_loadu_si256((const __m256i*)(a+i+j)) );
  }
}

static
void div_t2_t_test( uint8_t * poly , int len ) {
  add_shr1( poly , len );
  add_shr2( poly , len );
  add_shr4( poly , len );
  for(int i=1;i<len;i<<=1 ) { add_shr_jbyte( poly , len , i ); }
}




///////////////////////////

static
void _madd_2bits_test( uint8_t *c , const uint8_t *a , uint8_t b , int len )
{
  __m256i b_0 = _mm256_sub_epi8(_mm256_setzero_si256(),_mm256_set1_epi8( b&1 ));
  __m256i b_1 = _mm256_sub_epi8(_mm256_setzero_si256(),_mm256_set1_epi8( (b>>1)&1 ));

  __m256i _a0 = _mm256_loadu_si256((const __m256i*)a);
  __m256i a15 = _mm256_permute4x64_epi64(_mm256_srli_si256(_a0,15),0x4f); // 01,00,11,11
  __m256i _a_1 = _mm256_slli_si256(_a0,1)|a15;
  _mm256_storeu_si256((__m256i*)c, _mm256_loadu_si256((const __m256i*)c)^(shl_1(_a0,_a_1)&b_1)^(_a0&b_0) );

  for(int i=32;i<len;i+=32) {
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a+i));
    __m256i a_1 = _mm256_loadu_si256((const __m256i*)(a+i-1));
    _mm256_storeu_si256((__m256i*)(c+i), _mm256_loadu_si256((const __m256i*)(c+i))^(shl_1(a0,a_1)&b_1)^(a0&b_0) );
  }
}

// static
// void mul_2bits_test( uint8_t *c , const uint8_t *a , uint8_t b , int len )
// {
//   for(int i=0;i<len;i++) c[i]=0;
//   _madd_2bits_test( c , a , b , len );
//   c[len] = (a[len-1]>>7)&(b>>1);
// }


//////////////////////////////////////

//#include "gf2x_karatsuba.h"

// 512x512bit multiplication performed by Karatsuba algorithm
// where a and b are considered as having 8 digits of size 64 bits.
//void gf2x_mul_base_pclmul(OUT uint64_t *c, IN const uint64_t *a, IN const uint64_t *b);


void rkara3_mul_1536(uint64_t *c, const uint64_t *a, const uint64_t *b)
{
#if 1
  const uint64_t * f0 = a;
  const uint64_t * f1 = a+8;
  const uint64_t * f2 = a+16;
  const uint64_t * g0 = b;
  const uint64_t * g1 = b+8;
  const uint64_t * g2 = b+16;

  uint64_t h0[16];
  uint64_t h1[16];
  uint64_t hinf[20]; hinf[16]=0; // for shr_1
  uint64_t ht[16];
  uint64_t ht1[16];

///
  uint64_t f012[8];
  uint64_t g012[8];
  add( (uint8_t*)f012 , (const uint8_t*)f0 , (const uint8_t*)f1 , 64 );
  add( (uint8_t*)f012 , (const uint8_t*)f012 , (const uint8_t*)f2 , 64 ); // f0+f1+f2
  add( (uint8_t*)g012 , (const uint8_t*)g0 , (const uint8_t*)g1 , 64 );
  add( (uint8_t*)g012 , (const uint8_t*)g012 , (const uint8_t*)g2 , 64 ); // g0+g1+g2

///
  uint64_t tf[8];
  uint64_t tg[8];
  shl_1_test( (uint8_t*)tf , (const uint8_t *)f1 , 64 );
  shl_2_test( (uint8_t*)ht , (const uint8_t *)f2 , 64 );
  add( (uint8_t*)tf , (const uint8_t*)tf , (const uint8_t*)ht , 64 );
  uint8_t tf1 = (f1[7]>>63)^(f2[7]>>62); // 2bits

  shl_1_test( (uint8_t*)tg , (const uint8_t *)g1 , 64 );
  shl_2_test( (uint8_t*)ht , (const uint8_t *)g2 , 64 );
  add( (uint8_t*)tg , (const uint8_t*)tg , (const uint8_t*)ht , 64 );
  uint8_t tg1 = (g1[7]>>63)^(g2[7]>>62); // 2bits

  uint8_t tf1_mul_tg1 = (tf1*(tg1&1))^(tf1*(tg1&2)); // 3bits

// ht
  uint64_t tmp_f[8];
  uint64_t tmp_g[8];

  add( (uint8_t*)tmp_f , (const uint8_t*)tf , (const uint8_t*)f0 , 64 );
  add( (uint8_t*)tmp_g , (const uint8_t*)tg , (const uint8_t*)g0 , 64 );
  gf2x_mul_base_pclmul( ht , tmp_f , tmp_g );
  _madd_2bits_test( (uint8_t*)(ht+8) , (const uint8_t*)tmp_f , tg1 , 64 );
  _madd_2bits_test( (uint8_t*)(ht+8) , (const uint8_t*)tmp_g , tf1 , 64 );
  uint8_t ht_high = tf1_mul_tg1 ^ ((tmp_f[7]>>63)&(tg1>>1)) ^ ((tmp_g[7]>>63)&(tf1>>1));

// ht1
  add( (uint8_t*)tmp_f , (const uint8_t*)tf , (const uint8_t*)f012 , 64 );
  add( (uint8_t*)tmp_g , (const uint8_t*)tg , (const uint8_t*)g012 , 64 );
  gf2x_mul_base_pclmul( ht1 , tmp_f , tmp_g );
  _madd_2bits_test( (uint8_t*)(ht1+8) , (const uint8_t*)tmp_f , tg1 , 64 );
  _madd_2bits_test( (uint8_t*)(ht1+8) , (const uint8_t*)tmp_g , tf1 , 64 );
  uint8_t ht1_high = tf1_mul_tg1 ^ ((tmp_f[7]>>63)&(tg1>>1)) ^ ((tmp_g[7]>>63)&(tf1>>1));

// ht_ht1n
  add( (uint8_t*)ht1 , (const uint8_t*)ht1 , (const uint8_t*)ht , 128 );
  ht1_high ^= ht_high;  // 1bit

// h1
  gf2x_mul_base_pclmul( h1 , f012 , g012 );
// h0
  gf2x_mul_base_pclmul( h0 , f0 , g0 );
// hinf
  gf2x_mul_base_pclmul( hinf , f2 , g2 );

// U
  cpy( (uint8_t*)c , (const uint8_t*)h0 , 64 );
  add( (uint8_t*)(c+8) , (const uint8_t*)h0 , (const uint8_t*)h1 , 64*2 );
  add( (uint8_t*)(c+8) , (const uint8_t*)(c+8) , (const uint8_t*)(h0+8) , 64 );

// V
  uint64_t V[24+4] = {0};
  cpy( (uint8_t*)V , (const uint8_t*)ht , 128 );
  V[16] = ht_high;
  add( (uint8_t*)(V+8) , (const uint8_t*)(V+8) , (const uint8_t*)ht1 , 128 );
//  V[24] = ht1_high;
  uint8_t v_high = ht1_high;

  uint64_t tmp[16];
  shl_1_test( (uint8_t*)tmp , (const uint8_t*)ht1 , 128 ); // (ht+ht1)<<1
  add( (uint8_t*)V , (const uint8_t*)V , (const uint8_t*)tmp , 128 );
  V[16] ^= ((ht1[15]>>63)^(ht1_high<<1));

// W
  add( (uint8_t*)V , (const uint8_t*)V , (const uint8_t*)c , 192 );

  shl_2_test( (uint8_t*)tmp , (const uint8_t*)hinf , 128 );
  uint64_t hinf_shl2_high = (hinf[15]>>62);
  uint64_t tmp2[16];
  shr_1_test( (uint8_t*)tmp2 , (const uint8_t*)hinf , 128 );
  add( (uint8_t*)tmp , (const uint8_t*)tmp , (const uint8_t*)tmp2 , 128 );

  shr_2_test( (uint8_t*)V , (const uint8_t*)V , 192 );
  V[23] ^= (((uint64_t)v_high)<<62);

  add( (uint8_t*)V , (const uint8_t*)V , (const uint8_t*)tmp , 128 );
  V[16] ^= hinf_shl2_high;

  div_t2_t_test( (uint8_t*)V , 192 );

// output
  for(int i=0;i<8;i++) c[24+i]=0;
  cpy( (uint8_t*)(c+32) , (const uint8_t*)hinf , 128 );
  add( (uint8_t*)(c+8) , (const uint8_t*)(c+8) , (const uint8_t*)hinf , 128 );
  add( (uint8_t*)(c+16) , (const uint8_t*)(c+16) , (const uint8_t*)V , 192 );
  add( (uint8_t*)(c+8) , (const uint8_t*)(c+8) , (const uint8_t*)V , 192 );

#else
  //gf2x_mul_1536( c , a , b );
#endif
}


///////////////




void rkara3_mul_12288(uint64_t *c, const uint64_t *a, const uint64_t *b)
{
#if 1
  const uint64_t * f0 = a;
  const uint64_t * f1 = a+64;
  const uint64_t * f2 = a+128;
  const uint64_t * g0 = b;
  const uint64_t * g1 = b+64;
  const uint64_t * g2 = b+128;

  uint64_t h0[128];
  uint64_t h1[128];
  uint64_t hinf[128+4]; hinf[128]=0; // for shr_1
  uint64_t ht[128];
  uint64_t ht1[128];

///
  uint64_t f012[64];
  uint64_t g012[64];
  add( (uint8_t*)f012 , (const uint8_t*)f0 , (const uint8_t*)f1 , 512 );
  add( (uint8_t*)f012 , (const uint8_t*)f012 , (const uint8_t*)f2 , 512 ); // f0+f1+f2
  add( (uint8_t*)g012 , (const uint8_t*)g0 , (const uint8_t*)g1 , 512 );
  add( (uint8_t*)g012 , (const uint8_t*)g012 , (const uint8_t*)g2 , 512 ); // g0+g1+g2

///
  uint64_t tf[64];
  uint64_t tg[64];
  shl_1_test( (uint8_t*)tf , (const uint8_t *)f1 , 512 );
  shl_2_test( (uint8_t*)ht , (const uint8_t *)f2 , 512 );
  add( (uint8_t*)tf , (const uint8_t*)tf , (const uint8_t*)ht , 512 );
  uint8_t tf1 = (f1[64-1]>>63)^(f2[64-1]>>62); // 2bits

  shl_1_test( (uint8_t*)tg , (const uint8_t *)g1 , 512 );
  shl_2_test( (uint8_t*)ht , (const uint8_t *)g2 , 512 );
  add( (uint8_t*)tg , (const uint8_t*)tg , (const uint8_t*)ht , 512 );
  uint8_t tg1 = (g1[64-1]>>63)^(g2[64-1]>>62); // 2bits

  uint8_t tf1_mul_tg1 = (tf1*(tg1&1))^(tf1*(tg1&2)); // 3bits

// ht
  uint64_t tmp_f[64];
  uint64_t tmp_g[64];

  add( (uint8_t*)tmp_f , (const uint8_t*)tf , (const uint8_t*)f0 , 512 );
  add( (uint8_t*)tmp_g , (const uint8_t*)tg , (const uint8_t*)g0 , 512 );
  gf2x_mul_4096( ht , tmp_f , tmp_g );
  _madd_2bits_test( (uint8_t*)(ht+64) , (const uint8_t*)tmp_f , tg1 , 512 );
  _madd_2bits_test( (uint8_t*)(ht+64) , (const uint8_t*)tmp_g , tf1 , 512 );
  uint8_t ht_high = tf1_mul_tg1 ^ ((tmp_f[64-1]>>63)&(tg1>>1)) ^ ((tmp_g[64-1]>>63)&(tf1>>1));

// ht1
  add( (uint8_t*)tmp_f , (const uint8_t*)tf , (const uint8_t*)f012 , 512 );
  add( (uint8_t*)tmp_g , (const uint8_t*)tg , (const uint8_t*)g012 , 512 );
  gf2x_mul_4096( ht1 , tmp_f , tmp_g );
  _madd_2bits_test( (uint8_t*)(ht1+64) , (const uint8_t*)tmp_f , tg1 , 512 );
  _madd_2bits_test( (uint8_t*)(ht1+64) , (const uint8_t*)tmp_g , tf1 , 512 );
  uint8_t ht1_high = tf1_mul_tg1 ^ ((tmp_f[64-1]>>63)&(tg1>>1)) ^ ((tmp_g[64-1]>>63)&(tf1>>1));

// ht_ht1n
  add( (uint8_t*)ht1 , (const uint8_t*)ht1 , (const uint8_t*)ht , 1024 );
  ht1_high ^= ht_high;  // 1bit

// h1
  gf2x_mul_4096( h1 , f012 , g012 );
// h0
  gf2x_mul_4096( h0 , f0 , g0 );
// hinf
  gf2x_mul_4096( hinf , f2 , g2 );

// U
  cpy( (uint8_t*)c , (const uint8_t*)h0 , 512 );
  add( (uint8_t*)(c+64) , (const uint8_t*)h0 , (const uint8_t*)h1 , 512*2 );
  add( (uint8_t*)(c+64) , (const uint8_t*)(c+64) , (const uint8_t*)(h0+64) , 512 );

// V
  uint64_t V[192+4] = {0};
  cpy( (uint8_t*)V , (const uint8_t*)ht , 1024 );
  V[128] = ht_high;
  add( (uint8_t*)(V+64) , (const uint8_t*)(V+64) , (const uint8_t*)ht1 , 1024 );
//  V[24] = ht1_high;
  uint8_t v_high = ht1_high;

  uint64_t tmp[128];
  shl_1_test( (uint8_t*)tmp , (const uint8_t*)ht1 , 1024 ); // (ht+ht1)<<1
  add( (uint8_t*)V , (const uint8_t*)V , (const uint8_t*)tmp , 1024 );
  V[128] ^= ((ht1[128-1]>>63)^(ht1_high<<1));

// W
  add( (uint8_t*)V , (const uint8_t*)V , (const uint8_t*)c , 1536 );

  shl_2_test( (uint8_t*)tmp , (const uint8_t*)hinf , 1024 );
  uint64_t hinf_shl2_high = (hinf[128-1]>>62);
  uint64_t tmp2[128];
  shr_1_test( (uint8_t*)tmp2 , (const uint8_t*)hinf , 1024 );
  add( (uint8_t*)tmp , (const uint8_t*)tmp , (const uint8_t*)tmp2 , 1024 );

  shr_2_test( (uint8_t*)V , (const uint8_t*)V , 1536 );
  V[192-1] ^= (((uint64_t)v_high)<<62);

  add( (uint8_t*)V , (const uint8_t*)V , (const uint8_t*)tmp , 1024 );
  V[128] ^= hinf_shl2_high;

  div_t2_t_test( (uint8_t*)V , 1536 );

// output
  for(int i=0;i<64;i++) c[192+i]=0;
  cpy( (uint8_t*)(c+256) , (const uint8_t*)hinf , 1024 );
  add( (uint8_t*)(c+64) , (const uint8_t*)(c+64) , (const uint8_t*)hinf , 1024 );
  add( (uint8_t*)(c+128) , (const uint8_t*)(c+128) , (const uint8_t*)V , 1536 );
  add( (uint8_t*)(c+64) , (const uint8_t*)(c+64) , (const uint8_t*)V , 1536 );

#else
  //gf2x_mul_12288( c , a , b );
#endif
}


////////////////////////



void rkara3_mul_24576(uint64_t *c, const uint64_t *a, const uint64_t *b)
{
#if 1
  const uint64_t * f0 = a;
  const uint64_t * f1 = a+128;
  const uint64_t * f2 = a+256;
  const uint64_t * g0 = b;
  const uint64_t * g1 = b+128;
  const uint64_t * g2 = b+256;

  uint64_t h0[256];
  uint64_t h1[256];
  uint64_t hinf[256+4]; hinf[256]=0; // for shr_1
  uint64_t ht[256];
  uint64_t ht1[256];

///
  uint64_t f012[128];
  uint64_t g012[128];
  add( (uint8_t*)f012 , (const uint8_t*)f0 , (const uint8_t*)f1 , 1024 );
  add( (uint8_t*)f012 , (const uint8_t*)f012 , (const uint8_t*)f2 , 1024 ); // f0+f1+f2
  add( (uint8_t*)g012 , (const uint8_t*)g0 , (const uint8_t*)g1 , 1024 );
  add( (uint8_t*)g012 , (const uint8_t*)g012 , (const uint8_t*)g2 , 1024 ); // g0+g1+g2

///
  uint64_t tf[128];
  uint64_t tg[128];
  shl_1_test( (uint8_t*)tf , (const uint8_t *)f1 , 1024 );
  shl_2_test( (uint8_t*)ht , (const uint8_t *)f2 , 1024 );
  add( (uint8_t*)tf , (const uint8_t*)tf , (const uint8_t*)ht , 1024 );
  uint8_t tf1 = (f1[128-1]>>63)^(f2[128-1]>>62); // 2bits

  shl_1_test( (uint8_t*)tg , (const uint8_t *)g1 , 1024 );
  shl_2_test( (uint8_t*)ht , (const uint8_t *)g2 , 1024 );
  add( (uint8_t*)tg , (const uint8_t*)tg , (const uint8_t*)ht , 1024 );
  uint8_t tg1 = (g1[128-1]>>63)^(g2[128-1]>>62); // 2bits

  uint8_t tf1_mul_tg1 = (tf1*(tg1&1))^(tf1*(tg1&2)); // 3bits

// ht
  uint64_t tmp_f[128];
  uint64_t tmp_g[128];

  add( (uint8_t*)tmp_f , (const uint8_t*)tf , (const uint8_t*)f0 , 1024 );
  add( (uint8_t*)tmp_g , (const uint8_t*)tg , (const uint8_t*)g0 , 1024 );
  gf2x_mul_8192( ht , tmp_f , tmp_g );
  _madd_2bits_test( (uint8_t*)(ht+128) , (const uint8_t*)tmp_f , tg1 , 1024 );
  _madd_2bits_test( (uint8_t*)(ht+128) , (const uint8_t*)tmp_g , tf1 , 1024 );
  uint8_t ht_high = tf1_mul_tg1 ^ ((tmp_f[128-1]>>63)&(tg1>>1)) ^ ((tmp_g[128-1]>>63)&(tf1>>1));

// ht1
  add( (uint8_t*)tmp_f , (const uint8_t*)tf , (const uint8_t*)f012 , 1024 );
  add( (uint8_t*)tmp_g , (const uint8_t*)tg , (const uint8_t*)g012 , 1024 );
  gf2x_mul_8192( ht1 , tmp_f , tmp_g );
  _madd_2bits_test( (uint8_t*)(ht1+128) , (const uint8_t*)tmp_f , tg1 , 1024 );
  _madd_2bits_test( (uint8_t*)(ht1+128) , (const uint8_t*)tmp_g , tf1 , 1024 );
  uint8_t ht1_high = tf1_mul_tg1 ^ ((tmp_f[128-1]>>63)&(tg1>>1)) ^ ((tmp_g[128-1]>>63)&(tf1>>1));

// ht_ht1n
  add( (uint8_t*)ht1 , (const uint8_t*)ht1 , (const uint8_t*)ht , 2048 );
  ht1_high ^= ht_high;  // 1bit

// h1
  gf2x_mul_8192( h1 , f012 , g012 );
// h0
  gf2x_mul_8192( h0 , f0 , g0 );
// hinf
  gf2x_mul_8192( hinf , f2 , g2 );

// U
  cpy( (uint8_t*)c , (const uint8_t*)h0 , 1024 );
  add( (uint8_t*)(c+128) , (const uint8_t*)h0 , (const uint8_t*)h1 , 1024*2 );
  add( (uint8_t*)(c+128) , (const uint8_t*)(c+128) , (const uint8_t*)(h0+128) , 1024 );

// V
  uint64_t V[384+4] = {0};
  cpy( (uint8_t*)V , (const uint8_t*)ht , 2048 );
  V[256] = ht_high;
  add( (uint8_t*)(V+128) , (const uint8_t*)(V+128) , (const uint8_t*)ht1 , 2048 );
//  V[24] = ht1_high;
  uint8_t v_high = ht1_high;

  uint64_t tmp[256];
  shl_1_test( (uint8_t*)tmp , (const uint8_t*)ht1 , 2048 ); // (ht+ht1)<<1
  add( (uint8_t*)V , (const uint8_t*)V , (const uint8_t*)tmp , 2048 );
  V[256] ^= ((ht1[256-1]>>63)^(ht1_high<<1));

// W
  add( (uint8_t*)V , (const uint8_t*)V , (const uint8_t*)c , 3072 );

  shl_2_test( (uint8_t*)tmp , (const uint8_t*)hinf , 2048 );
  uint64_t hinf_shl2_high = (hinf[256-1]>>62);
  uint64_t tmp2[256];
  shr_1_test( (uint8_t*)tmp2 , (const uint8_t*)hinf , 2048 );
  add( (uint8_t*)tmp , (const uint8_t*)tmp , (const uint8_t*)tmp2 , 2048 );

  shr_2_test( (uint8_t*)V , (const uint8_t*)V , 3072 );
  V[384-1] ^= (((uint64_t)v_high)<<62);

  add( (uint8_t*)V , (const uint8_t*)V , (const uint8_t*)tmp , 2048 );
  V[256] ^= hinf_shl2_high;

  div_t2_t_test( (uint8_t*)V , 3072 );

// output
  for(int i=0;i<128;i++) c[384+i]=0;
  cpy( (uint8_t*)(c+512) , (const uint8_t*)hinf , 2048 );
  add( (uint8_t*)(c+128) , (const uint8_t*)(c+128) , (const uint8_t*)hinf , 2048 );
  add( (uint8_t*)(c+256) , (const uint8_t*)(c+256) , (const uint8_t*)V , 3072 );
  add( (uint8_t*)(c+128) , (const uint8_t*)(c+128) , (const uint8_t*)V , 3072 );

#else
  //gf2x_mul_24576( c , a , b );
#endif
}


/////////////////////
#define gfmul_1(O,A,B) karat_mult_1((__m128i*)O, (const __m128i*)A, (const __m128i*)B);


void karat_mult_1(__m128i *C, const __m128i *A, const __m128i *B) {
	__m128i D1[2];
	__m128i D0[2], D2[2];
	__m128i Al = _mm_loadu_si128(A);
	__m128i Ah = _mm_loadu_si128(A + 1);
	__m128i Bl = _mm_loadu_si128(B);
	__m128i Bh = _mm_loadu_si128(B + 1);

	//	Compute Al.Bl=D0
	__m128i DD0 = _mm_clmulepi64_si128(Al, Bl, 0);//a0*b0
	__m128i DD2 = _mm_clmulepi64_si128(Al, Bl, 0x11);//a1*b1 1
	__m128i AAlpAAh = _mm_xor_si128(Al, _mm_shuffle_epi32(Al, 0x4e));//shuffle 0x4e : a1a0->a0a1 (a3,a4:64bit)
	__m128i BBlpBBh = _mm_xor_si128(Bl, _mm_shuffle_epi32(Bl, 0x4e)); //3
	__m128i DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));//a1b0^a0b1=(a0^a1)(b0^b1)^a0b0^a1b1
	D0[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));//DD1을
	D0[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));//DD0, DD2의 중간에 xor하는 느낌

	//	Compute Ah.Bh=D2
	DD0 = _mm_clmulepi64_si128(Ah, Bh, 0);//a2*b2
	DD2 = _mm_clmulepi64_si128(Ah, Bh, 0x11);//a3*b3 
	AAlpAAh = _mm_xor_si128(Ah, _mm_shuffle_epi32(Ah, 0x4e));//shuffle 0x4e : a3a2->a2a3 (a2,a3:64bit)
	BBlpBBh = _mm_xor_si128(Bh, _mm_shuffle_epi32(Bh, 0x4e));//xor_si128 : xor
	DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D2[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D2[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	// Compute AlpAh.BlpBh=D1
	// Initialisation of AlpAh and BlpBh
	__m128i AlpAh = _mm_xor_si128(Al,Ah);
	__m128i BlpBh = _mm_xor_si128(Bl,Bh);
	DD0 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0);
	DD2 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0x11);
	AAlpAAh = _mm_xor_si128(AlpAh, _mm_shuffle_epi32(AlpAh, 0x4e));
	BBlpBBh = _mm_xor_si128(BlpBh, _mm_shuffle_epi32(BlpBh, 0x4e));
	DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
	D1[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
	D1[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

	
	__m128i middle = _mm_xor_si128(D0[1], D2[0]);
	C[0] = D0[0];
	C[1] = middle ^ D0[0] ^ D1[0];
	C[2] = middle ^ D1[1] ^ D2[1];
	C[3] = D2[1];
}

#define gfmul_2(O,A,B) karat2_mult_2((__m256i*)O, (const __m256i*)A, (const __m256i*)B);
static inline void karat2_mult_2(__m256i *C, const __m256i *A,const __m256i *B)
{
  __m256i D0[2], D1[2], D2[2], SAA, SBB;
  gfmul_1(D0, A, B);
  gfmul_1(D2, (A + 1), (B + 1));

  SAA = A[0] ^ A[1];
  SBB = B[0] ^ B[1];

  gfmul_1(D1, &SAA, &SBB);

  __m256i middle = _mm256_xor_si256(D0[1], D2[0]);
  C[0] = D0[0];
  C[1] = middle ^ D0[0] ^ D1[0];
  C[2] = middle ^ D1[1] ^ D2[1];
  C[3] = D2[1];

}

#define gfmul_6(O,A,B) karat3_mult_6((__m256i *)O,(const __m256i *)A,(const __m256i *)B);

static inline void karat3_mult_6(__m256i *Out, const  __m256i *A,  const __m256i *B){
 const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
 static __m256i middle;
 static __m256i aa01[2], bb01[2], aa02[2], bb02[2], aa12[2], bb12[2];
 static __m256i D0[4], D1[4], D2[4], D3[4], D4[4], D5[4];
 a0 = A;
 a1 = A + 2;
 a2 = A + 4;
 b0 = B;
 b1 = B + 2;
 b2 = B + 4;
 for (int16_t i = 0; i < 2; i++)
 {
  aa01[i] = a0[i] ^ a1[i];
  bb01[i] = b0[i] ^ b1[i];
  aa12[i] = a2[i] ^ a1[i];
  bb12[i] = b2[i] ^ b1[i];
  aa02[i] = a0[i] ^ a2[i];
  bb02[i] = b0[i] ^ b2[i];
 }
 gfmul_2(D3, aa01, bb01);
 gfmul_2(D4, aa02, bb02);
 gfmul_2(D5, aa12, bb12);
 gfmul_2(D0, a0, b0);
 gfmul_2(D1, a1, b1);
 gfmul_2(D2, a2, b2);
 for (int16_t i = 0; i < 2; i++)
 {
  int16_t j = i + 2;
  middle = D0[i] ^ D1[i] ^ D0[j];
  Out[i] = D0[i];
  Out[j] = D3[i] ^ middle;
  Out[j + 2] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
  middle = D1[j] ^ D2[i] ^ D2[j];
  Out[j + 4] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
  Out[i + 8] = D5[j] ^ middle;
  Out[j + 8] = D2[j];
 }
}

#define gfmul_12(O,A,B) karat2_mult_12((__m256i *)O,(const __m256i *)A,(const __m256i *)B);

static inline void karat2_mult_12(__m256i *C, const  __m256i *A,  const __m256i *B){
 static __m256i D0[12], D1[12], D2[12], SAA[6], SBB[6];
 gfmul_6(D0, A, B);
 gfmul_6(D2, (A+6), (B+6));
 for(int32_t i = 0; i < 6; i++) {
  int32_t is = i + 6;
  SAA[i] = A[i] ^ A[is];  SBB[i] = B[i] ^ B[is]; }
 gfmul_6(D1, SAA, SBB);
 for(int32_t i = 0; i < 6; i++) {
  int32_t is = i + 6;
  int32_t is2 = is + 6;
  int32_t is3 = is2 + 6;
  __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
  C[i]   = D0[i];
  C[is]  = middle ^ D0[i] ^ D1[i];
  C[is2] = middle ^ D1[is] ^ D2[is];
  C[is3] = D2[is];
 }
}



#define gfmul_24(O,A,B) karat2_mult_24((__m256i *)O,(const __m256i *)A,(const __m256i *)B);

static inline void karat2_mult_24(__m256i *C, const  __m256i *A, const  __m256i *B){
 static __m256i D0[24], D1[24], D2[24], SAA[12], SBB[12];
 gfmul_12(D0, A, B);
 gfmul_12(D2, (A+12), (B+12));
 for(int32_t i = 0; i < 12; i++) {
  int32_t is = i + 12;
  SAA[i] = A[i] ^ A[is];  SBB[i] = B[i] ^ B[is]; }
 gfmul_12(D1, SAA, SBB);
 for(int32_t i = 0; i < 12; i++) {
  int32_t is = i + 12;
  int32_t is2 = is + 12;
  int32_t is3 = is2 + 12;
  __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
  C[i]   = D0[i];
  C[is]  = middle ^ D0[i] ^ D1[i];
  C[is2] = middle ^ D1[is] ^ D2[is];
  C[is3] = D2[is];
 }
}


#define gfmul_48(O,A,B) karat2_mult_48((__m256i *)O,(const __m256i *)A,(const __m256i *)B);

static inline void karat2_mult_48(__m256i *C, const  __m256i *A,  const __m256i *B){
 static __m256i D0[48], D1[48], D2[48], SAA[24], SBB[24];
 gfmul_24(D0, A, B);
 gfmul_24(D2, (A+24), (B+24));
 for(int32_t i = 0; i < 24; i++) {
  int32_t is = i + 24;
  SAA[i] = A[i] ^ A[is];  SBB[i] = B[i] ^ B[is]; }
 gfmul_24(D1, SAA, SBB);
 for(int32_t i = 0; i < 24; i++) {
  int32_t is = i + 24;
  int32_t is2 = is + 24;
  int32_t is3 = is2 + 24;
  __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
  C[i]   = D0[i];
  C[is]  = middle ^ D0[i] ^ D1[i];
  C[is2] = middle ^ D1[is] ^ D2[is];
  C[is3] = D2[is];
 }
}

#define gfmul_96(O,A,B) karat2_mult_96((__m256i *)O,(const  __m256i *)A,(const  __m256i *)B);

static inline void karat2_mult_96(__m256i *C, const    __m256i *A,  const   __m256i *B){
 static __m256i D0[96], D1[96], D2[96], SAA[48], SBB[48];
 gfmul_48(D0, A, B);
 gfmul_48(D2, (A+48), (B+48));
 for(int32_t i = 0; i < 48; i++) {
  int32_t is = i + 48;
  SAA[i] = A[i] ^ A[is];  SBB[i] = B[i] ^ B[is]; }
 gfmul_48(D1, SAA, SBB);
 for(int32_t i = 0; i < 48; i++) {
  int32_t is = i + 48;
  int32_t is2 = is + 48;
  int32_t is3 = is2 + 48;
  __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
  C[i]   = D0[i];
  C[is]  = middle ^ D0[i] ^ D1[i];
  C[is2] = middle ^ D1[is] ^ D2[is];
  C[is3] = D2[is];
 }
}

static inline void divide_by_x_plus_one_256(__m256i *in, __m256i *out, int32_t size){
    out[0] = in[0];
    for(int32_t i = 1; i < size; i++) {
        out[i] = _mm256_xor_si256(out[i - 1], in[i]);
    }
}

static inline void divide_by_x_plus_one_128(__m256i *out, __m256i *in, int32_t size){
    __m128i *A = (__m128i *) in;
    __m128i *B = (__m128i *) out;

    B[0] = A[0];
    for(int32_t i = 1; i < size; i++) {
        B[i] = _mm_xor_si128(B[i - 1], A[i]);
    }
}

static inline void divide_by_x_plus_one_64(__m256i *out, __m256i *in, int32_t size){
    uint64_t *A = (uint64_t*) in;
    uint64_t *B = (uint64_t*) out;

    B[0] = A[0];
    for(int32_t i = 1; i < size; i++) {
        B[i]= B[i - 1] ^ A[i];
    }
}
//len = 1: karat_karat_PCLMULQDQ
static inline void gfmul_1_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m128i *A128 = (const __m128i *)A, *B128 = (const __m128i *)B;
    __m128i *Out128 = (__m128i *)Out;
    __m128i D1[2];
    __m128i D0[2], D2[2];
    __m128i Al = _mm_loadu_si128(A128);
    __m128i Ah = _mm_loadu_si128(A128 + 1);
    __m128i Bl = _mm_loadu_si128(B128);
    __m128i Bh = _mm_loadu_si128(B128 + 1);

    //	Compute Al.Bl=D0
    __m128i DD0 = _mm_clmulepi64_si128(Al, Bl, 0);
    __m128i DD2 = _mm_clmulepi64_si128(Al, Bl, 0x11);
    __m128i AAlpAAh = _mm_xor_si128(Al, _mm_shuffle_epi32(Al, 0x4e));
    __m128i BBlpBBh = _mm_xor_si128(Bl, _mm_shuffle_epi32(Bl, 0x4e)); 
    __m128i DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
    D0[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
    D0[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

    //	Compute Ah.Bh=D2
    DD0 = _mm_clmulepi64_si128(Ah, Bh, 0);
    DD2 = _mm_clmulepi64_si128(Ah, Bh, 0x11);
    AAlpAAh = _mm_xor_si128(Ah, _mm_shuffle_epi32(Ah, 0x4e));
    BBlpBBh = _mm_xor_si128(Bh, _mm_shuffle_epi32(Bh, 0x4e));
    DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
    D2[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
    D2[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

    // Compute AlpAh.BlpBh=D1
    // Initialisation of AlpAh and BlpBh
    __m128i AlpAh = _mm_xor_si128(Al,Ah);
    __m128i BlpBh = _mm_xor_si128(Bl,Bh);
    DD0 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0);
    DD2 = _mm_clmulepi64_si128(AlpAh, BlpBh, 0x11);
    AAlpAAh = _mm_xor_si128(AlpAh, _mm_shuffle_epi32(AlpAh, 0x4e));
    BBlpBBh = _mm_xor_si128(BlpBh, _mm_shuffle_epi32(BlpBh, 0x4e));
    DD1 = _mm_xor_si128(_mm_xor_si128(DD0, DD2), _mm_clmulepi64_si128(AAlpAAh, BBlpBBh, 0));
    D1[0] = _mm_xor_si128(DD0, _mm_unpacklo_epi64(_mm_setzero_si128(), DD1));
    D1[1] = _mm_xor_si128(DD2, _mm_unpackhi_epi64(DD1, _mm_setzero_si128()));

    __m128i middle = _mm_xor_si128(D0[1], D2[0]);
    Out128[0] = D0[0];
    Out128[1] = middle ^ D0[0] ^ D1[0];
    Out128[2] = middle ^ D1[1] ^ D2[1];
    Out128[3] = D2[1];
}

//len = 2: 2-Karatsuba
static inline void gfmul_2_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[2], D1[2], D2[2], SAA[1], SBB[1];
    gfmul_1_pclmul(D0, A, B);
    gfmul_1_pclmul(D2, (A+1), (B+1));
    SAA[0] = A[0] ^ A[1];
    SBB[0] = B[0] ^ B[1];
    gfmul_1_pclmul(D1, SAA, SBB);
    __m256i middle = _mm256_xor_si256(D0[1], D2[0]);
    Out[0] = D0[0];
    Out[1] = middle ^ D0[0] ^ D1[0];
    Out[2] = middle ^ D1[1] ^ D2[1];
    Out[3] = D2[1];
}

//len = 3: 2-Karatsuba
static inline void gfmul_3_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[4], D1[4], D2[2], SAA[2], SBB[2];
    gfmul_2_pclmul(D0, A, B);
    gfmul_1_pclmul(D2, (A+2), (B+2));
    SAA[0] = A[0] ^ A[2];
    SBB[0] = B[0] ^ B[2];
    SAA[1] = A[1];
    SBB[1] = B[1];
    gfmul_2_pclmul(D1, SAA, SBB);
    __m256i middle = _mm256_xor_si256(D0[2], D2[0]);
    Out[0]         = D0[0];
    Out[2]         = middle ^ D0[0] ^ D1[0];
    Out[4]         = middle ^ D1[2];
    middle         = _mm256_xor_si256(D0[3], D2[1]);
    Out[1]         = D0[1];
    Out[3]         = middle ^ D0[1] ^ D1[1];
    Out[5]         = middle ^ D1[3];
}

//len = 4: 2-Karatsuba
static inline void gfmul_4_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[4], D1[4], D2[4], SAA[2], SBB[2];
    gfmul_2_pclmul(D0, A, B);
    gfmul_2_pclmul(D2, (A+2), (B+2));
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 2;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_2_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 2;
        int32_t is2 = is + 2;
        int32_t is3 = is2 + 2;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 5: 2-Karatsuba
static inline void gfmul_5_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[6], D1[6], D2[4], SAA[3], SBB[3];
    gfmul_3_pclmul(D0, A, B);
    gfmul_2_pclmul(D2, (A+3), (B+3));
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 3;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    SAA[2]=A[2];        
    SBB[2]=B[2];    
    gfmul_3_pclmul(D1, SAA, SBB);
    __m256i middle = _mm256_xor_si256(D0[3], D2[0]);
    Out[0] = D0[0];
    Out[3] = middle ^ D0[0] ^ D1[0];
    Out[6] = middle ^ D1[3] ^ D2[3];
    Out[9] = D2[3];

    for(int32_t i = 1; i < 3; i++) {
        int32_t is = i + 3;
        int32_t is2 = is + 3;
        middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 6: 2-Karatsuba
static inline void gfmul_6_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[6], D1[6], D2[6], SAA[3], SBB[3];
    gfmul_3_pclmul(D0, A, B);
    gfmul_3_pclmul(D2, (A+3), (B+3));
    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 3;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_3_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 3;
        int32_t is2 = is + 3;
        int32_t is3 = is2 + 3;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 7: 2-Karatsuba
static inline void gfmul_7_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[8], D1[8], D2[6], SAA[4], SBB[4];
    gfmul_4_pclmul(D0, A, B);
    gfmul_3_pclmul(D2, (A+4), (B+4));
    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 4;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[3]=A[3];        SBB[3]=B[3];    gfmul_4_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 2; i++) {
        int32_t is = i + 4;
        int32_t is2 = is + 4;
        int32_t is3 = is2 + 4;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 2; i < 4; i++) {
        int32_t is = i + 4;
        int32_t is2 = is + 4;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 8: 2-Karatsuba
static inline void gfmul_8_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[8], D1[8], D2[8], SAA[4], SBB[4];
    gfmul_4_pclmul(D0, A, B);
    gfmul_4_pclmul(D2, (A+4), (B+4));
    for(int32_t i = 0; i < 4; i++) {
        int32_t is = i + 4;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_4_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 4; i++) {
        int32_t is = i + 4;
        int32_t is2 = is + 4;
        int32_t is3 = is2 + 4;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 9: 2-Karatsuba
static inline void gfmul_9_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[10], D1[10], D2[8], SAA[5], SBB[5];
    gfmul_5_pclmul(D0, A, B);
    gfmul_4_pclmul(D2, (A+5), (B+5));
    for(int32_t i = 0; i < 4; i++) {
        int32_t is = i + 5;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[4]=A[4];        SBB[4]=B[4];    gfmul_5_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 3; i++) {
        int32_t is = i + 5;
        int32_t is2 = is + 5;
        int32_t is3 = is2 + 5;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 3; i < 5; i++) {
        int32_t is = i + 5;
        int32_t is2 = is + 5;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 11: 2-Karatsuba
static inline void gfmul_11_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[12], D1[12], D2[10], SAA[6], SBB[6];
    gfmul_6_pclmul(D0, A, B);
    gfmul_5_pclmul(D2, (A+6), (B+6));
    for(int32_t i = 0; i < 5; i++) {
        int32_t is = i + 6;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[5]=A[5];        SBB[5]=B[5];    gfmul_6_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 4; i++) {
        int32_t is = i + 6;
        int32_t is2 = is + 6;
        int32_t is3 = is2 + 6;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 4; i < 6; i++) {
        int32_t is = i + 6;
        int32_t is2 = is + 6;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 12: 3-Karatsuba
static inline void gfmul_12_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    const __m256i *a0, *a1, *a2, *b0, *b1, *b2;
    __m256i middle;
    static __m256i aa01[4], bb01[4], aa02[4], bb02[4], aa12[4], bb12[4];
    static __m256i D0[8], D1[8], D2[8], D3[8], D4[8], D5[8];
    a0 = A;
    a1 = A + 4;
    a2 = A + 8;
    b0 = B;
    b1 = B + 4;
    b2 = B + 8;
    for (int16_t i = 0; i < 4; i++)
    {
        aa01[i] = a0[i] ^ a1[i];
        bb01[i] = b0[i] ^ b1[i];
        aa12[i] = a2[i] ^ a1[i];
        bb12[i] = b2[i] ^ b1[i];
        aa02[i] = a0[i] ^ a2[i];
        bb02[i] = b0[i] ^ b2[i];
    }
    gfmul_4_pclmul(D3, aa01, bb01);
    gfmul_4_pclmul(D4, aa02, bb02);
    gfmul_4_pclmul(D5, aa12, bb12);
    gfmul_4_pclmul(D0, a0, b0);
    gfmul_4_pclmul(D1, a1, b1);
    gfmul_4_pclmul(D2, a2, b2);
    for (int16_t i = 0; i < 4; i++)
    {
        int16_t j = i + 4;
        middle = D0[i] ^ D1[i] ^ D0[j];
        Out[i] = D0[i];
        Out[j] = D3[i] ^ middle;
        Out[j + 4] = D4[i] ^ D2[i] ^ D3[j] ^ D1[j] ^ middle;
        middle = D1[j] ^ D2[i] ^ D2[j];
        Out[j + 8] = D5[i] ^ D4[j] ^ D0[j] ^ D1[i] ^ middle;
        Out[i + 16] = D5[j] ^ middle;
        Out[j + 16] = D2[j];
    }
}

//len = 13: 2-Karatsuba
static inline void gfmul_13_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[14], D1[14], D2[12], SAA[7], SBB[7];
    gfmul_7_pclmul(D0, A, B);
    gfmul_6_pclmul(D2, (A+7), (B+7));
    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 7;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[6]=A[6];        SBB[6]=B[6];    gfmul_7_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 5; i++) {
        int32_t is = i + 7;
        int32_t is2 = is + 7;
        int32_t is3 = is2 + 7;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 5; i < 7; i++) {
        int32_t is = i + 7;
        int32_t is2 = is + 7;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 14: 2-Karatsuba
static inline void gfmul_14_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[14], D1[14], D2[14], SAA[7], SBB[7];
    gfmul_7_pclmul(D0, A, B);
    gfmul_7_pclmul(D2, (A+7), (B+7));
    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 7;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_7_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 7;
        int32_t is2 = is + 7;
        int32_t is3 = is2 + 7;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 15: 2-Karatsuba
static inline void gfmul_15_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[16], D1[16], D2[14], SAA[8], SBB[8];
    gfmul_8_pclmul(D0, A, B);
    gfmul_7_pclmul(D2, (A+8), (B+8));
    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 8;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[7]=A[7];        SBB[7]=B[7];    gfmul_8_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 6; i++) {
        int32_t is = i + 8;
        int32_t is2 = is + 8;
        int32_t is3 = is2 + 8;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 6; i < 8; i++) {
        int32_t is = i + 8;
        int32_t is2 = is + 8;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 16: 2-Karatsuba
static inline void gfmul_16_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[16], D1[16], D2[16], SAA[8], SBB[8];
    gfmul_8_pclmul(D0, A, B);
    gfmul_8_pclmul(D2, (A+8), (B+8));
    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 8;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_8_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 8;
        int32_t is2 = is + 8;
        int32_t is3 = is2 + 8;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 17: 2-Karatsuba
static inline void gfmul_17_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[18], D1[18], D2[16], SAA[9], SBB[9];
    gfmul_9_pclmul(D0, A, B);
    gfmul_8_pclmul(D2, (A+9), (B+9));
    for(int32_t i = 0; i < 8; i++) {
        int32_t is = i + 9;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[8]=A[8];        SBB[8]=B[8];    gfmul_9_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 7; i++) {
        int32_t is = i + 9;
        int32_t is2 = is + 9;
        int32_t is3 = is2 + 9;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 7; i < 9; i++) {
        int32_t is = i + 9;
        int32_t is2 = is + 9;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 18: TC3_256
static inline void gfmul_18_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[12], W1[15], W2[16], W3[16], W4[12], tmp[16];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[6];
    U2 = (const __m256i *)&A256[12];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[6];
    V2 = (const __m256i *)&B256[12];
    for (int32_t i = 0 ; i < 6 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_6_pclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 6 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    W0[7] = U2[5];
    W4[7] = V2[5];
    for (int32_t i = 0 ; i < 6 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[6] = W0[6];
    W3[7] = W0[7];
    W2[6] = W4[6];
    W2[7] = W4[7];
    for (int32_t i = 0 ; i < 6 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_8_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_8_pclmul(W2, W0, W4);
    gfmul_6_pclmul(W4, U2, V2);
    gfmul_6_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 12 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 11 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[11] = W2[12];
    W2[12] = W2[13];
    W2[13] = W2[14];
    W2[14] = W2[15];
    for (int32_t i = 0 ; i < 12 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 12 ; i < 15 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[15] = W3[15];
    for (int32_t i = 0 ; i < 12 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 16);
    for (int32_t i = 0 ; i < 11 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[11] = W3[12];
    tmp[12] = W3[13];
    tmp[13] = W3[14];
    tmp[14] = W3[15];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 15);
    for (int32_t i = 0 ; i < 12 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 12 ; i < 15 ; i++) {
        W1[i] = W2[i];
    }
    for (int32_t i = 0 ; i < 14 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 6; i++) {
        int32_t j = i + 6;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 6] = W1[j] ^ W2[i];
        Out[j + 12] = W2[j] ^ W3[i];
        Out[i + 24] = W3[j] ^ W4[i];
        Out[j + 24] = W4[j];
    }
    Out[18] ^= W1[12];
    Out[19] ^= W1[13];
    Out[20] ^= W1[14];
    Out[24] ^= W2[12];
    Out[25] ^= W2[13];
    Out[26] ^= W2[14];
    Out[30] ^= W3[12];
    Out[31] ^= W3[13];
}

//len = 19: TC3_256
static inline void gfmul_19_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[14], W1[15], W2[16], W3[16], W4[10], tmp[16];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[7];
    U2 = (const __m256i *)&A256[14];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[7];
    V2 = (const __m256i *)&B256[14];
    for (int32_t i = 0 ; i < 5 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[5] = U0[5] ^ U1[5];
    W2[5] = V0[5] ^ V1[5];
    W3[6] = U0[6] ^ U1[6];
    W2[6] = V0[6] ^ V1[6];
    gfmul_7_pclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 6 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    
    W0[7] = U1[6];
    W4[7] = V1[6];
    for (int32_t i = 0 ; i < 7 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[7] = W0[7];
    W2[7] = W4[7];
    for (int32_t i = 0 ; i < 7 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_8_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_8_pclmul(W2, W0, W4);
    gfmul_5_pclmul(W4, U2, V2);
    gfmul_7_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 16 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 14 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 13 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[13] = W2[14];
    W2[14] = W2[15];
    for (int32_t i = 0 ; i < 10 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 10 ; i < 15 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[15] = W3[15];
    for (int32_t i = 0 ; i < 10 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 16);
    for (int32_t i = 0 ; i < 13 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[13] = W3[14];
    tmp[14] = W3[15];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 15);
    for (int32_t i = 0 ; i < 10 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 10 ; i < 14 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[14] = W2[14];
    for (int32_t i = 0 ; i < 14 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 3; i++) {
        int32_t j = i + 7;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 7] = W1[j] ^ W2[i];
        Out[j + 14] = W2[j] ^ W3[i];
        Out[i + 28] = W3[j] ^ W4[i];
        Out[j + 28] = W4[j];
    }
    for (int32_t i = 3; i < 7; i++) {
        int32_t j = i + 7;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 7] = W1[j] ^ W2[i];
        Out[j + 14] = W2[j] ^ W3[i];
        Out[i + 28] = W3[j] ^ W4[i];
    }
    Out[21] ^= W1[14];
    Out[28] ^= W2[14];
}

//len = 24: 2-Karatsuba
static inline void gfmul_24_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[24], D1[24], D2[24], SAA[12], SBB[12];
    gfmul_12_pclmul(D0, A, B);
    gfmul_12_pclmul(D2, (A+12), (B+12));
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 12;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_12_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 12;
        int32_t is2 = is + 12;
        int32_t is3 = is2 + 12;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 25: 2-Karatsuba
static inline void gfmul_25_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[26], D1[26], D2[24], SAA[13], SBB[13];
    gfmul_13_pclmul(D0, A, B);
    gfmul_12_pclmul(D2, (A+13), (B+13));
    for(int32_t i = 0; i < 12; i++) {
        int32_t is = i + 13;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[12]=A[12];        SBB[12]=B[12];    gfmul_13_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 11; i++) {
        int32_t is = i + 13;
        int32_t is2 = is + 13;
        int32_t is3 = is2 + 13;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 11; i < 13; i++) {
        int32_t is = i + 13;
        int32_t is2 = is + 13;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 31: 2-Karatsuba
static inline void gfmul_31_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[32], D1[32], D2[30], SAA[16], SBB[16];
    gfmul_16_pclmul(D0, A, B);
    gfmul_15_pclmul(D2, (A+16), (B+16));
    for(int32_t i = 0; i < 15; i++) {
        int32_t is = i + 16;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[15]=A[15];        SBB[15]=B[15];    gfmul_16_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 14; i++) {
        int32_t is = i + 16;
        int32_t is2 = is + 16;
        int32_t is3 = is2 + 16;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 14; i < 16; i++) {
        int32_t is = i + 16;
        int32_t is2 = is + 16;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 32: 2-Karatsuba
static inline void gfmul_32_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[32], D1[32], D2[32], SAA[16], SBB[16];
    gfmul_16_pclmul(D0, A, B);
    gfmul_16_pclmul(D2, (A+16), (B+16));
    for(int32_t i = 0; i < 16; i++) {
        int32_t is = i + 16;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_16_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 16; i++) {
        int32_t is = i + 16;
        int32_t is2 = is + 16;
        int32_t is3 = is2 + 16;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 33: 2-Karatsuba
static inline void gfmul_33_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[34], D1[34], D2[32], SAA[17], SBB[17];
    gfmul_17_pclmul(D0, A, B);
    gfmul_16_pclmul(D2, (A+17), (B+17));
    for(int32_t i = 0; i < 16; i++) {
        int32_t is = i + 17;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[16]=A[16];        SBB[16]=B[16];    gfmul_17_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 15; i++) {
        int32_t is = i + 17;
        int32_t is2 = is + 17;
        int32_t is3 = is2 + 17;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 15; i < 17; i++) {
        int32_t is = i + 17;
        int32_t is2 = is + 17;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 34: TC3_128
static inline void gfmul_34_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    static __m256i U0[12], U1[12], U2[11], V0[12], V1[12], V2[11];
    static __m256i W0[24], W1[24], W2[24], W3[25], W4[22];
    static __m256i tmp[25];
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    for(int32_t i = 0; i < 11; i++) {
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 46]));
        V1[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 46]));
        U2[i]= _mm256_lddqu_si256((__m256i const *)(& A[i4 + 92]));
        V2[i]= _mm256_lddqu_si256((__m256i const *)(& B[i4 + 92]));
    }
    U0[11]= (__m256i){A[44], A[45], 0x0ul, 0x0ul};
    V0[11]= (__m256i){B[44], B[45], 0x0ul, 0x0ul};
    U1[11]= (__m256i){A[90], A[91], 0x0ul, 0x0ul};
    V1[11]= (__m256i){B[90], B[91], 0x0ul, 0x0ul};
    for (int32_t i = 0 ; i < 11 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[11] = U0[11] ^ U1[11];
    W2[11] = V0[11] ^ V1[11];
    gfmul_12_pclmul(W1, W2, W3);
    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *V1_64 = ((uint64_t *) V1);
    W0[0] = (__m256i){0ul, 0ul, U1_64[0], U1_64[1]};
    W4[0] = (__m256i){0ul, 0ul, V1_64[0], V1_64[1]};
    U1_64 = ((uint64_t *) U1) + 2;
    V1_64 = ((uint64_t *) V1) + 2;
    for(int32_t i = 0; i < 11; i++) {
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_lddqu_si256((__m256i const *)(& V1_64[i4]));
        W4[i1] ^= V2[i];
    }
    for (int32_t i = 0 ; i < 12 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0 ; i < 12 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_12_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 24; i++) {
        W3[i] = tmp[i];
    }
    gfmul_12_pclmul(W2, W0, W4);
    gfmul_11_pclmul(W4, U2, V2);
    gfmul_12_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 24 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 23 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_64 = ((uint64_t *) W2) + 2;
    uint64_t * U2_64 = ((uint64_t *) W0) + 2;
    for(int32_t i = 0; i < 22; i++) {
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    W2[22]=(__m256i){U2_64[88], U2_64[89], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[88]));
    W2[23]=(__m256i){U1_64[92], U1_64[93], 0ul, 0ul};
    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ (__m256i){0x0ul, 0x0ul, U1_64[0], U1_64[1]};
    U1_64 = ((uint64_t *) W4) + 2;
    for(int32_t i = 2; i < 22; i++) {
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[i4-8]));
    }
    tmp[22] = W2[22] ^ W3[22] ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[80]));
    tmp[23] = W2[23] ^ W3[23] ^ (__m256i){U1_64[84],U1_64[85], 0ul, 0ul};
    divide_by_x_plus_one_128(W2, tmp, 48);
    U1_64 = ((uint64_t *) W3) + 2;
    U2_64 = ((uint64_t *) W1) + 2;
    for(int32_t i = 0; i < 22; i++) {
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i const *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i const *)(& U2_64[i4]));
    }
    tmp[22]=(__m256i){U2_64[88], U2_64[89], 0ul, 0ul} ^ _mm256_lddqu_si256((__m256i const *)(& U1_64[88]));
    tmp[23]=(__m256i){U1_64[92], U1_64[93], 0ul, 0ul};
    divide_by_x_plus_one_128(W3, tmp, 47);
    for (int32_t i = 0 ; i < 22 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[22] ^= W2[22];
    W1[23] = W2[23];
    for (int32_t i = 0 ; i < 23 ; i++) {
        W2[i] ^= W3[i];
    }
    for(int32_t i = 0; i < 22; i++) {
        Out[i] = W0[i];
        Out[i + 23] = W2[i];
        Out[i + 46] = W4[i];
    }
    Out[22] = W0[22];
    Out[45] = W2[22];
    Out[46] ^= W2[23];
    U1_64 = ((uint64_t *) &Out[11]) + 2;
    U2_64 = ((uint64_t *) &Out[34]) + 2;
    __m256i aux;
    for(int32_t i = 0; i < 24; i++) {
        int32_t i4 = i << 2;
        aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);
    }
}

//len = 40: TC3_256
static inline void gfmul_40_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[28], W1[29], W2[30], W3[30], W4[24], tmp[30];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[14];
    U2 = (const __m256i *)&A256[28];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[14];
    V2 = (const __m256i *)&B256[28];
    for (int32_t i = 0 ; i < 12 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[12] = U0[12] ^ U1[12];
    W2[12] = V0[12] ^ V1[12];
    W3[13] = U0[13] ^ U1[13];
    W2[13] = V0[13] ^ V1[13];
    gfmul_14_pclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 13 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    
    W0[14] = U1[13];
    W4[14] = V1[13];
    for (int32_t i = 0 ; i < 14 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[14] = W0[14];
    W2[14] = W4[14];
    for (int32_t i = 0 ; i < 14 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_15_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 30 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_15_pclmul(W2, W0, W4);
    gfmul_12_pclmul(W4, U2, V2);
    gfmul_14_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 30 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 28 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 27 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[27] = W2[28];
    W2[28] = W2[29];
    for (int32_t i = 0 ; i < 24 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 24 ; i < 29 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[29] = W3[29];
    for (int32_t i = 0 ; i < 24 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 30);
    for (int32_t i = 0 ; i < 27 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[27] = W3[28];
    tmp[28] = W3[29];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 29);
    for (int32_t i = 0 ; i < 24 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 24 ; i < 28 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[28] = W2[28];
    for (int32_t i = 0 ; i < 28 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 10; i++) {
        int32_t j = i + 14;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 14] = W1[j] ^ W2[i];
        Out[j + 28] = W2[j] ^ W3[i];
        Out[i + 56] = W3[j] ^ W4[i];
        Out[j + 56] = W4[j];
    }
    for (int32_t i = 10; i < 14; i++) {
        int32_t j = i + 14;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 14] = W1[j] ^ W2[i];
        Out[j + 28] = W2[j] ^ W3[i];
        Out[i + 56] = W3[j] ^ W4[i];
    }
    Out[42] ^= W1[28];
    Out[56] ^= W2[28];
}

//len = 54: TC3_128
static inline void gfmul_54_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[36], W1[38], W2[38], W3[38], W4[36];
    static __m256i tmp[38];
    __m128i zero128;
    zero128 = _mm_setzero_si128();
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[18];
    U2 = (const __m256i *)&A256[36];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[18];
    V2 = (const __m256i *)&B256[36];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_18_pclmul(W1, W2, W3);
    const __m128i *U1_128 = ((const __m128i *) U1);
    const __m128i *V1_128 = ((const __m128i *) V1);
    W0[0] = _mm256_set_m128i(U1_128[0],zero128);
    W4[0] = _mm256_set_m128i(V1_128[0],zero128);
    U1_128 = ((const __m128i *) U1) + 1;
    V1_128 = ((const __m128i *) V1) + 1;
    for(int32_t i = 0; i < 17; i++) {
        int32_t i2 = i << 1;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W0[i1] ^= U2[i];
        W4[i1] = _mm256_set_m128i(V1_128[i2+1],V1_128[i2]);
        W4[i1] ^= V2[i];
    }
    W0[18] = _mm256_set_m128i(zero128,U1_128[34]) ^ U2[17];
    W4[18] = _mm256_set_m128i(zero128,V1_128[34]) ^ V2[17];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[18] = W0[18];
    W2[18] = W4[18];
    for (int32_t i = 0 ; i < 18 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_19_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 38; i++) {
        W3[i] = tmp[i];
    }
    gfmul_19_pclmul(W2, W0, W4);
    gfmul_18_pclmul(W4, U2, V2);
    gfmul_18_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 38 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 36 ; i++) {
        W1[i] ^= W0[i];
    }
    U1_128 = ((const __m128i *) W2) + 1;
    const __m128i * U2_128 = ((__m128i *) W0) + 1;
    for(int32_t i = 0; i < 35; i++) {
        int32_t i2 = i << 1;
        W2[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]);
        W2[i] ^= _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    W2[35]=_mm256_set_m128i(U1_128[71],U2_128[70]^U1_128[70]);
    W2[36]=_mm256_set_m128i(U1_128[73],U1_128[72]);
    W2[37]=_mm256_set_m128i(zero128,U1_128[74]);
    U1_128 = ((__m128i *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0];
    tmp[1] = W2[1] ^ W3[1] ^ W4[1] ^ _mm256_set_m128i(U1_128[0],zero128);
    U1_128 = ((const __m128i *) W4) + 1;
    for(int32_t i = 2; i < 36; i++) {
        int32_t i2 = i << 1;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^_mm256_set_m128i(U1_128[i2 - 3],U1_128[i2 - 4]);
    }
    tmp[36] = W2[36] ^ W3[36] ^ _mm256_set_m128i(U1_128[69],U1_128[68]);
    tmp[37] = W2[37] ^ W3[37] ^ _mm256_set_m128i(zero128,U1_128[70]);
    divide_by_x_plus_one_128(W2, tmp, 76);
    U1_128 = ((const __m128i *) W3) + 1;
    U2_128 = ((const __m128i *) W1) + 1;
    for(int32_t i = 0; i < 35; i++) {
        int32_t i2 = i << 1;
        tmp[i] = _mm256_set_m128i(U1_128[i2+1],U1_128[i2]) ^ _mm256_set_m128i(U2_128[i2+1],U2_128[i2]);
    }
    tmp[35]=_mm256_set_m128i(U1_128[71], U2_128[70]^U1_128[70]);
    tmp[36]=_mm256_set_m128i(U1_128[73],U1_128[72]);
    tmp[37]=_mm256_set_m128i(zero128, U1_128[74]);
    divide_by_x_plus_one_128(W3, tmp, 75);
    for (int32_t i = 0 ; i < 36 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    W1[36] = W2[36];
    W1[37] = W2[37];
    for (int32_t i = 0 ; i < 37 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 18; i++)
    {
        int32_t j = i + 18;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 18] = W1[j] ^ W2[i];
        Out[j + 36] = W2[j] ^ W3[i];
        Out[i + 72] = W3[j] ^ W4[i];
        Out[j + 72] = W4[j];
    }
    Out[54] ^= W1[36];
    Out[55] ^= W1[37];
    Out[72] ^= W2[36];
    Out[73] ^= W2[37];
    Out[90] ^= W3[36];
}

//len = 80: 2-Karatsuba
static inline void gfmul_80_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[80], D1[80], D2[80], SAA[40], SBB[40];
    gfmul_40_pclmul(D0, A, B);
    gfmul_40_pclmul(D2, (A+40), (B+40));
    for(int32_t i = 0; i < 40; i++) {
        int32_t is = i + 40;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_40_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 40; i++) {
        int32_t is = i + 40;
        int32_t is2 = is + 40;
        int32_t is3 = is2 + 40;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//////////////////////////////

//len = 48: 2-Karatsuba
void gfmul_48_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[48], D1[48], D2[48], SAA[24], SBB[24];
    gfmul_24_pclmul(D0, A, B);
    gfmul_24_pclmul(D2, (A+24), (B+24));
    for(int32_t i = 0; i < 24; i++) {
        int32_t is = i + 24;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_24_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 24; i++) {
        int32_t is = i + 24;
        int32_t is2 = is + 24;
        int32_t is3 = is2 + 24;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 49: 2-Karatsuba
void gfmul_49_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[50], D1[50], D2[48], SAA[25], SBB[25];
    gfmul_25_pclmul(D0, A, B);
    gfmul_24_pclmul(D2, (A+25), (B+25));
    for(int32_t i = 0; i < 24; i++) {
        int32_t is = i + 25;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
        SAA[24]=A[24];        SBB[24]=B[24];    gfmul_25_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 23; i++) {
        int32_t is = i + 25;
        int32_t is2 = is + 25;
        int32_t is3 = is2 + 25;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
    for(int32_t i = 23; i < 25; i++) {
        int32_t is = i + 25;
        int32_t is2 = is + 25;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is];
    }
}

//len = 96: TC3_256
void gfmul_96_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[64], W1[67], W2[68], W3[68], W4[64], tmp[68];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[32];
    U2 = (const __m256i *)&A256[64];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[32];
    V2 = (const __m256i *)&B256[64];
    for (int32_t i = 0 ; i < 32 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_32_pclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 32 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    W0[33] = U2[31];
    W4[33] = V2[31];
    for (int32_t i = 0 ; i < 32 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[32] = W0[32];
    W3[33] = W0[33];
    W2[32] = W4[32];
    W2[33] = W4[33];
    for (int32_t i = 0 ; i < 32 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_34_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 68 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_34_pclmul(W2, W0, W4);
    gfmul_32_pclmul(W4, U2, V2);
    gfmul_32_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 68 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 64 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 63 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[63] = W2[64];
    W2[64] = W2[65];
    W2[65] = W2[66];
    W2[66] = W2[67];
    for (int32_t i = 0 ; i < 64 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 64 ; i < 67 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[67] = W3[67];
    for (int32_t i = 0 ; i < 64 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 68);
    for (int32_t i = 0 ; i < 63 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[63] = W3[64];
    tmp[64] = W3[65];
    tmp[65] = W3[66];
    tmp[66] = W3[67];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 67);
    for (int32_t i = 0 ; i < 64 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 64 ; i < 67 ; i++) {
        W1[i] = W2[i];
    }
    for (int32_t i = 0 ; i < 66 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 32; i++) {
        int32_t j = i + 32;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 32] = W1[j] ^ W2[i];
        Out[j + 64] = W2[j] ^ W3[i];
        Out[i + 128] = W3[j] ^ W4[i];
        Out[j + 128] = W4[j];
    }
    Out[96] ^= W1[64];
    Out[97] ^= W1[65];
    Out[98] ^= W1[66];
    Out[128] ^= W2[64];
    Out[129] ^= W2[65];
    Out[130] ^= W2[66];
    Out[160] ^= W3[64];
    Out[161] ^= W3[65];
}

//len = 97: TC3_256
void gfmul_97_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    const __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[66], W1[67], W2[68], W3[68], W4[62], tmp[68];
    static   __m256i zero = {0ul, 0ul, 0ul, 0ul};
    U0 = (const __m256i *)&A256[0];
    U1 = (const __m256i *)&A256[33];
    U2 = (const __m256i *)&A256[66];
    V0 = (const __m256i *)&B256[0];
    V1 = (const __m256i *)&B256[33];
    V2 = (const __m256i *)&B256[66];
    for (int32_t i = 0 ; i < 31 ; i++) {
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    W3[31] = U0[31] ^ U1[31];
    W2[31] = V0[31] ^ V1[31];
    W3[32] = U0[32] ^ U1[32];
    W2[32] = V0[32] ^ V1[32];
    gfmul_33_pclmul(W1, W2, W3);
    W0[0] = zero;
    W4[0] = zero;
    W0[1] = U1[0];
    W4[1] = V1[0];
    for (int32_t i = 1 ; i < 32 ; i++) {
        W0[i + 1] = U1[i] ^ U2[i - 1];
        W4[i + 1] = V1[i] ^ V2[i - 1];
    }
    
    W0[33] = U1[32];
    W4[33] = V1[32];
    for (int32_t i = 0 ; i < 33 ; i++) {
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    W3[33] = W0[33];
    W2[33] = W4[33];
    for (int32_t i = 0 ; i < 33 ; i++) {
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_34_pclmul(tmp, W3, W2);
    for (int32_t i = 0 ; i < 68 ; i++) {
        W3[i] = tmp[i];
    }
    gfmul_34_pclmul(W2, W0, W4);
    gfmul_31_pclmul(W4, U2, V2);
    gfmul_33_pclmul(W0, U0, V0);
    for (int32_t i = 0 ; i < 68 ; i++) {
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0 ; i < 66 ; i++) {
        W1[i] ^= W0[i];
    }
    for (int32_t i = 0 ; i < 65 ; i++) {
        int32_t i1 = i + 1;
        W2[i] = W2[i1] ^ W0[i1];
    }
    W2[65] = W2[66];
    W2[66] = W2[67];
    for (int32_t i = 0 ; i < 62 ; i++) {
        tmp[i] = W2[i] ^ W3[i] ^ W4[i];
    }
    for (int32_t i = 62 ; i < 67 ; i++) {
        tmp[i] = W2[i] ^ W3[i];
    }
    tmp[67] = W3[67];
    for (int32_t i = 0 ; i < 62 ; i++) {
        tmp[i + 3] ^= W4[i];
    }
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W2, 68);
    for (int32_t i = 0 ; i < 65 ; i++) {
        int32_t i1 = i + 1;
        tmp[i] = W3[i1] ^ W1[i1];
    }
    tmp[65] = W3[66];
    tmp[66] = W3[67];
    divide_by_x_plus_one_256((__m256i   *)tmp, (__m256i   *)W3, 67);
    for (int32_t i = 0 ; i < 62 ; i++) {
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 62 ; i < 66 ; i++) {
        W1[i] ^= W2[i];
    }
    W1[66] = W2[66];
    for (int32_t i = 0 ; i < 66 ; i++) {
        W2[i] ^= W3[i];
    }
    for (int32_t i = 0; i < 29; i++) {
        int32_t j = i + 33;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 33] = W1[j] ^ W2[i];
        Out[j + 66] = W2[j] ^ W3[i];
        Out[i + 132] = W3[j] ^ W4[i];
        Out[j + 132] = W4[j];
    }
    for (int32_t i = 29; i < 33; i++) {
        int32_t j = i + 33;
        Out[i] = W0[i];
        Out[j] = W0[j] ^ W1[i];
        Out[j + 33] = W1[j] ^ W2[i];
        Out[j + 66] = W2[j] ^ W3[i];
        Out[i + 132] = W3[j] ^ W4[i];
    }
    Out[99] ^= W1[66];
    Out[132] ^= W2[66];
}

//len = 160: 2-Karatsuba
void gfmul_160_pclmul(__m256i *Out,  const __m256i *A,  const __m256i *B){
    static __m256i D0[160], D1[160], D2[160], SAA[80], SBB[80];
    gfmul_80_pclmul(D0, A, B);
    gfmul_80_pclmul(D2, (A+80), (B+80));
    for(int32_t i = 0; i < 80; i++) {
        int32_t is = i + 80;
        SAA[i] = A[i] ^ A[is];
        SBB[i] = B[i] ^ B[is];
    }
    gfmul_80_pclmul(D1, SAA, SBB);
    for(int32_t i = 0; i < 80; i++) {
        int32_t is = i + 80;
        int32_t is2 = is + 80;
        int32_t is3 = is2 + 80;
        __m256i middle = _mm256_xor_si256(D0[is], D2[i]);
        Out[i]   = D0[i];
        Out[is]  = middle ^ D0[i] ^ D1[i];
        Out[is2] = middle ^ D1[is] ^ D2[is];
        Out[is3] = D2[is];
    }
}

//len = 161: TC3_64
void gfmul_161_pclmul(__m256i *Out,  const __m256i *A256,  const __m256i *B256){
    static __m256i UV[324];
    static __m256i *U0, *U1, *U2, *V0, *V1, *V2;
    static __m256i W0[108], W1[108], W2[108], W3[108], W4[108];
    static   __m256i tmp[322];
    const uint64_t *A = (const uint64_t *) A256;
    const uint64_t *B = (const uint64_t *) B256;
    U0 = UV;
    U1 = U0 + 54;
    U2 = U1 + 54;
    V0 = U2 + 54;
    V1 = V0 + 54;
    V2 = V1 + 54;
    for (int32_t i = 0; i < 53; i++){
        int32_t i4 = i << 2;
        U0[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4]));
        V0[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4]));
        U1[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4 + 215]));
        V1[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4 + 215]));
        U2[i]= _mm256_lddqu_si256((const __m256i   *)(& A[i4 + 430]));
        V2[i]= _mm256_lddqu_si256((const __m256i   *)(& B[i4 + 430]));
    }
    U0[53]= (__m256i){A[212], A[213], A[214], 0x0ul};
    V0[53]= (__m256i){B[212], B[213], B[214], 0x0ul};
    U1[53]= (__m256i){A[427], A[428], A[429], 0x0ul};
    V1[53]= (__m256i){B[427], B[428], B[429], 0x0ul};
    U2[53]= (__m256i){A[642], A[643], 0x0ul, 0x0ul};
    V2[53]= (__m256i){B[642], B[643], 0x0ul, 0x0ul};
    for (int32_t i = 0; i < 54; i++){
        W3[i] = U0[i] ^ U1[i] ^ U2[i];
        W2[i] = V0[i] ^ V1[i] ^ V2[i];
    }
    gfmul_54_pclmul(W1, W2, W3);
    uint64_t *U1_64 = ((uint64_t *) U1);
    uint64_t *U2_64 = ((uint64_t *) U2);
    uint64_t *V1_64 = ((uint64_t *) V1);
    uint64_t *V2_64 = ((uint64_t *) V2);
    W0[0] = (__m256i){0ul, U1_64[0], U1_64[1] ^ U2_64[0], U1_64[2] ^ U2_64[1]};
    W4[0] = (__m256i){0ul, V1_64[0], V1_64[1] ^ V2_64[0], V1_64[2] ^ V2_64[1]};
    U1_64 = ((uint64_t *) U1) + 3;
    U2_64 = ((uint64_t *) U2) + 2;
    V1_64 = ((uint64_t *) V1) + 3;
    V2_64 = ((uint64_t *) V2) + 2;
    for (int32_t i = 0; i < 53; i++){
        int32_t i4 = i << 2;
        int32_t i1 = i + 1;
        W0[i1] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));
        W0[i1] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
        W4[i1] = _mm256_lddqu_si256((__m256i   *)(& V1_64[i4]));
        W4[i1] ^= _mm256_lddqu_si256((__m256i   *)(& V2_64[i4]));
    }
    for (int32_t i = 0; i < 54; i++){
        W3[i] ^= W0[i];
        W2[i] ^= W4[i];
    }
    for (int32_t i = 0; i < 54; i++){
        W0[i] ^= U0[i];
        W4[i] ^= V0[i];
    }
    gfmul_54_pclmul(tmp, W3, W2);
    for (int32_t i = 0; i < 108; i++){
        W3[i] = tmp[i];
    }
    gfmul_54_pclmul(W2, W0, W4);
    gfmul_54_pclmul(W4, U2, V2);
    gfmul_54_pclmul(W0, U0, V0);
    for (int32_t i = 0; i < 108; i++){
        W3[i] ^= W2[i];
    }
    for (int32_t i = 0; i < 108; i++){
        W1[i] ^= W0[i];
    }
    U1_64 = ((uint64_t *) W2) + 1;
    U2_64 = ((uint64_t *) W0) + 1;
    for (int32_t i = 0; i < 107; i++){
        int32_t i4 = i << 2;
        W2[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4]));
        W2[i] ^= _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
    }
    W2[107]=(__m256i){U1_64[428]^U2_64[428], U1_64[429]^U2_64[429], U1_64[430]^U2_64[430], 0x0ul};
    U1_64 = ((uint64_t *) W4);
    tmp[0] = W2[0] ^ W3[0] ^ W4[0] ^ (__m256i){0x0ul, 0x0ul, 0x0ul, U1_64[0]};
    U1_64++;
    for (int32_t i = 1; i < 107; i++){
        int32_t i4 = i << 2;
        tmp[i] = W2[i] ^ W3[i] ^ W4[i] ^ _mm256_lddqu_si256((__m256i   *)(& U1_64[i4 - 4]));
    }
    tmp[107] = W2[107] ^ W3[107] ^ (__m256i){U1_64[424],U1_64[425],U1_64[426], 0x0ul};
    divide_by_x_plus_one_64(tmp, tmp, 432);
    for (int32_t i = 0; i < 108; i++){
        W2[i] = tmp[i];
    }
    U1_64 = ((uint64_t *) W3) + 1;
    U2_64 = ((uint64_t *) W1) + 1;
    for (int32_t i = 0; i < 107; i++){
        int32_t i4 = i << 2;
        tmp[i] = _mm256_lddqu_si256((__m256i   *)(& U1_64[i4])) ^ _mm256_lddqu_si256((__m256i   *)(& U2_64[i4]));
    }
    tmp[107]=(__m256i){U1_64[428]^U2_64[428],U1_64[429]^U2_64[429],U1_64[430]^U2_64[430],0x0ul};
    divide_by_x_plus_one_64(tmp, tmp, 431);
    for (int32_t i = 0; i < 108; i++){
        W3[i] = tmp[i];
    }
    for (int32_t i = 0; i < 108; i++){
        W1[i] ^= W2[i] ^ W4[i];
    }
    for (int32_t i = 0; i < 108; i++){
        W2[i] ^= W3[i];
    }
        memset((__m256i*)tmp, 0, sizeof(__m256i) * 322);
    for (int32_t i = 0; i < 107; i++){
        tmp[i] = W0[i];
        tmp[i + 215] = W4[i];
    }
        tmp[107] = W0[107];
    U1_64 = ((uint64_t *) &tmp[53]) + 3;
    U2_64 = ((uint64_t *) &tmp[107]) + 2;
    V1_64 = ((uint64_t *) &tmp[161]) + 1;
    for(int32_t i = 0; i < 108; i++) {
        int32_t i4 = i << 2;
        __m256i aux = _mm256_lddqu_si256 ((__m256i *) (& U1_64[i4])) ^ W1[i];
        _mm256_storeu_si256 ((__m256i *) (& U1_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& U2_64[i4])) ^ W2[i];
        _mm256_storeu_si256 ((__m256i *) (& U2_64[i4]), aux);
        aux = _mm256_lddqu_si256 ((__m256i *) (& V1_64[i4])) ^ W3[i];
        _mm256_storeu_si256 ((__m256i *) (& V1_64[i4]), aux);
    }
    for(int32_t i = 0; i < 322; i++) {
        Out[i] = tmp[i];
    }
}

static inline
void mul_64( uint8_t *c, const uint8_t *a, const uint8_t *b)
{
  __m128i a128 = _mm_loadu_si128((const __m128i*)a);
  __m128i b128 = _mm_loadu_si128((const __m128i*)b);
  __m128i c128 = _mm_clmulepi64_si128( a128 , b128 , 0 );
  _mm_storeu_si128((__m128i*)c,c128);
}

static inline
void madd_64( uint8_t *c, const uint8_t *a, const uint8_t* b, int len_b)
{
  __m128i a128 = _mm_loadu_si128((const __m128i*)a);
  __m128i carry = _mm_setzero_si128();

  for(int i=0;i<len_b;i+=16){
    __m128i b128 = _mm_loadu_si128((const __m128i*)(b+i));
    __m128i c128 = _mm_loadu_si128((const __m128i*)(c+i));
    __m128i c0 = _mm_clmulepi64_si128( b128 , a128 , 0 );
    __m128i c1 = _mm_clmulepi64_si128( b128 , a128 , 1 );
    c128 ^= c0^carry^_mm_slli_si128(c1,8);
    carry = _mm_srli_si128(c1,8);
    _mm_storeu_si128((__m128i*)(c+i),c128);
  }
  __m128i c128 = _mm_loadu_si128((const __m128i*)(c+len_b));
  c128 ^= carry;
  _mm_storeu_si128((__m128i*)(c+len_b),c128);
}

void rkara3_mul_12352(uint64_t *c, const uint64_t *a, const uint64_t *b)
{
  rkara3_mul_12288(c,a,b);
  mul_64( (uint8_t*)(c+384), (const uint8_t*)(a+192) , (const uint8_t*)(b+192) );
  madd_64( (uint8_t*)(c+192), (const uint8_t*)(a+192) , (const uint8_t*)b , 1536 );
  madd_64( (uint8_t*)(c+192), (const uint8_t*)(b+192) , (const uint8_t*)a , 1536 );
}

static inline
void madd_128( uint8_t *c, const uint8_t *a, const uint8_t* b, int len_b)
{
  __m128i a128 = _mm_loadu_si128((const __m128i*)a);
  __m128i carry = _mm_setzero_si128();

  for(int i=0;i<len_b;i+=16){
    __m128i b128 = _mm_loadu_si128((const __m128i*)(b+i));
    __m128i c128 = _mm_loadu_si128((const __m128i*)(c+i));
    __m128i c0 = _mm_clmulepi64_si128( b128 , a128 , 0 );
    __m128i c1 = _mm_clmulepi64_si128( b128 , a128 , 1 )^_mm_clmulepi64_si128( b128 , a128 , 0x10 );
    c128 ^= c0^carry^_mm_slli_si128(c1,8);
    carry = _mm_srli_si128(c1,8)^_mm_clmulepi64_si128( b128 , a128 , 0x11 );
    _mm_storeu_si128((__m128i*)(c+i),c128);
  }
  __m128i c128 = _mm_loadu_si128((const __m128i*)(c+len_b));
  c128 ^= carry;
  _mm_storeu_si128((__m128i*)(c+len_b),c128);
}


void rkara3_mul_24704(uint64_t *c, const uint64_t *a, const uint64_t *b)
{
#if 1
  rkara3_mul_24576(c,a,b);
  for(int i=0;i<4;i++) c[768+i] = 0;
  madd_128( (uint8_t*)(c+768), (const uint8_t*)(a+384) , (const uint8_t*)(b+384) , 16 );
  madd_128( (uint8_t*)(c+384), (const uint8_t*)(a+384) , (const uint8_t*)b , 3072 );
  madd_128( (uint8_t*)(c+384), (const uint8_t*)(b+384) , (const uint8_t*)a , 3072 );
#else
  gf2x_mul_24704(c,a,b);
#endif
}
# GF(2)[x] Polynomial Multiplication on AVX2
This directory contains code for generating and benchmarking optimized implementations of polynomial multiplication over GF(2)[x] using SIMD instruction sets, targeting AVX2 on x86 processors.

---

## Structure

```
AVX2/
├── gf2x_opt_AVX2/   # Code generator for optimizing GF(2)[x] multiplication
├── hqc_opt_AVX2/    # HQC integrated with our GF(2)[x] multiplication
└── bike_opt_AVX2/   # BIKE integrated with our GF(2)[x] multiplication
```

---

## Requirements

An x86 CPU with AVX2 and VPCLMULQDQ instruction support

---

## Build and Run

### 1. gf2x_opt_AVX2
#### 1) Optimize GF(2)[x] multiplication

Run the optimizer to search for the best multiplication algorithm for each degree:

```bash
cd GFmulOpt_AVX2
make gfopt
./gfopt_VPCLMUL
```
For the AVX2-PCLMULQDQ setting, run:
```bash
cd GFmulOpt_AVX2
make gfopt_PCLMUL
./gfopt_PCLMUL
```


This will generate benchmark files in `src/` (e.g., `gfopt_len_2.c`, `gfopt_len_3.c`, ...), each of which tests polynomial multiplication algorithms for a specific dimension.

You can tune the optimization via `gfopt.h`:
- `N_TEST` : Number of iterations for performance testing
- `MAX_LEN`: Maximum polynomial degree to optimize

To reset before re-running:

```bash
make clean
```

#### 2) Generate multiplication code

To generate code for specific degrees (e.g., 70, 141, 226):

```bash
make gfopt_codegen
./gfopt_codegen 70 141 226
```
For the AVX2-PCLMULQDQ setting, run:
```bash
make gfopt_codegen_PCLMUL
./gfopt_codegen_PCLMUL 70 141 226
```

Generated code will appear in the `result/` directory.

---
### 2. hqc_opt_AVX2

This directory contains code that integrates our optimized implementations of polynomial multiplication over GF(2)[x] on AVX2 into the HQC.

To evaluate performance, enter one of the parameter-specific subdirectories:

- `hqc-1/`
- `hqc-3/`
- `hqc-5/`

Then run:

```bash
make test_speed_all
make run_tests
```

These commands will build and run the following tests:

(1) Standard HQC with AVX2_PCLMULQDQ multiplication  
(2) HQC with base multiplication replaced by `SB_SB_256_VPCLMUL` (AVX2-VPCLMULQDQ)  
(3) HQC integrated with our optimized method on AVX2-PCLMULQDQ  
(4) HQC integrated with our optimized method on AVX2-VPCLMULQDQ  
(5) HQC integrated with the `gf2x` library  

To build individual tests manually:

```bash
make test/test_speed_pclmul          # (1)
make test/test_speed_vpclmul         # (2)
make test/test_speed_gfopt_pclmul    # (3)
make test/test_speed_gfopt_vpclmul   # (4)
make test/test_speed_gf2xlib         # (5)
```

To run them individually:

```bash
./test/test_speed_pclmul             # (1)
./test/test_speed_vpclmul            # (2)
./test/test_speed_gfopt_pclmul       # (3)
./test/test_speed_gfopt_vpclmul      # (4)
./test/test_speed_gf2xlib            # (5)
```
---
### 3. bike_opt_AVX2

This directory contains code that integrates our optimized polynomial multiplication into the BIKE post-quantum cryptographic scheme using AVX2.

To build and run a specific test manually, you can configure it using `cmake` options:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=<1|2|3> -DGFMUL_VER=<0|1|2|3> -DVPCLMUL=<0|1>" ..
make
./bike-test
```

- `LEVEL`:
  - `1`: BIKE security level 1
  - `3`: BIKE security level 3
  - `5`: BIKE security level 5

- `GFMUL_VER`:
  - `0`: Original multiplication in BIKE
  - `1`: Our proposed method (GFmulOpt)
  - `2`: method proposed in "Optimizing BIKE for the Intel Haswell and ARM Cortex-M4" by Ming-Shing Chen et al. (CHES 2021)
  - `3`: gf2x library (must be used with `-DVPCLMUL=0`)

- `VPCLMUL`:
  - `0`: Use AVX2-PCLMULQDQ
  - `1`: Use AVX2-VPCLMULQDQ

--Example--
To run BIKE level 3 with our optimized method using VPCLMULQDQ:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-DLEVEL=3 -DGFMUL_VER=1 -DVPCLMUL=1" ..
make
./bike-test
```

# SIMD-Optimized GF(2)[x] Polynomial Multiplication

This repository contains the source code accompanying a submission to the Journal of Cryptographic Engineering.

---

## Overview

This project provides optimized implementations of polynomial multiplication over GF(2)[x] using SIMD instruction sets, targeting:

- `AVX2` on x86 processors
- `NEON` on ARMv8 processors

It includes:

- A code generator that finds the most efficient combination of Karatsuba and Toom–Cook multiplication strategies.
- Integration with the HQC and BIKE post-quantum cryptography schemes to measure performance impact.
- Benchmarks comparing the proposed method with existing implementations, including the `gf2x` library.

---

##  Directory Structure

```
.
├── AVX2/                  # SIMD-based implementation for x86 (AVX2)
│   ├── hqc_opt_AVX2/         # HQC integrated with our optimized GF(2)[x] multiplication (AVX2)
│   ├── bike_opt_AVX2/        # BIKE integrated with our optimized GF(2)[x] multiplication (AVX2)
│   └── GFmulOpt_AVX2/        # GF(2)[x] multiplication optimizer for AVX2
│
└── NEON/                  # SIMD-based implementation for ARMv8 (NEON)
    ├── hqc_opt_NEON/         # HQC integrated with our optimized GF(2)[x] multiplication (NEON)
    ├── bike_opt_NEON/        # BIKE integrated with our optimized GF(2)[x] multiplication (NEON)
    ├── GFmulOpt_NEON/        # GF(2)[x] multiplication optimizer for NEON
    └── enable_ccr/           # Utility to enable access to the ARM cycle counter (PMCCNTR_EL0) for cycle-accurate benchmarking
```
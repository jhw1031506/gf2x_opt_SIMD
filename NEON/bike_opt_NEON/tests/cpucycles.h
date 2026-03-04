#ifndef CPUCYCLES_H
#define CPUCYCLES_H

#include <stdint.h>

static inline uint64_t cpucycles(void) {
  
    uint64_t val;
    __asm__ volatile("isb;mrs %0, pmccntr_el0" : "=r"(val));
    return val;
}


uint64_t cpucycles_overhead(void);

#endif



CC:=gcc
CFLAGS:=-O3 -std=c99 -funroll-all-loops  -pedantic -Wall -Wextra
CFLAGS_PCLMUL:=-O3 -std=c99 -funroll-all-loops  -pedantic -Wall -Wextra
DEFINES_PCLMUL:= -DENV_AVX2

AVX2_SUPPORTED    := $(shell cat /proc/cpuinfo | grep -q ' avx2' && echo y || echo n)
VPCLMUL_SUPPORTED := $(shell cat /proc/cpuinfo | grep -q ' vpclmulqdq' && echo y || echo n)
NEON_SUPPORTED    := $(shell cat /proc/cpuinfo | grep -q -E ' neon| asimd' && echo y || echo n)

ifeq ($(AVX2_SUPPORTED), y)
    CFLAGS += -msse2 -mavx2 -mpclmul
    DEFINES += -DENV_AVX2
	ifeq ($(VPCLMUL_SUPPORTED), y)
		CFLAGS += -mvpclmulqdq
		DEFINES += -DENV_VPCLMUL
		BEST_ALG_RESULT += ./best_alg/best_alg_result_AVX2_VPCLMUL.h
		TARGET:= gfopt_VPCLMUL
	else
		BEST_ALG_RESULT += ./best_alg/best_alg_result_AVX2_PCLMUL.h
		TARGET:= gfopt_PCLMUL
	endif
else
	ifeq ($(NEON_SUPPORTED), y)
    DEFINES += -DENV_NEON
	BEST_ALG_RESULT += ./best_alg/best_alg_result_NEON.h
	TARGET:= gfopt_NEON
	else
	BEST_ALG_RESULT += ./best_alg/best_alg_result_new.h
	TARGET:= gfopt_new
	endif
endif

gfopt: ./gfopt.c ./gfopt_head_init ./gfopt_tail $(BEST_ALG_RESULT) ./gfopt_tail_basemul ./tools.c ./gfopt.h
	rm -f ./src/*
	$(CC) $(CFLAGS) $(DEFINES) ./tools.c ./gfopt.c -o $(TARGET)

gfopt_codegen: ./tools.c $(BEST_ALG_RESULT) ./gfopt.h ./gfopt_codegen.c result/result_head result/result_tail
	$(CC) $(CFLAGS) $(DEFINES) ./tools.c ./gfopt_codegen.c -o gfopt_codegen


basemul_test: ./basemul_test.c
	$(CC) $(CFLAGS) $(DEFINES) ./basemul_test.c -o basemul_test

clean: 
	rm -f ./src/*
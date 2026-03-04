CC:=gcc

CFLAGS:=-O3 -std=c99 -funroll-all-loops -flto -mavx -mavx2 -mpclmul -pedantic -Wall -Wextra
CFLAGS_VPCLMUL:=-O3 -std=c99 -funroll-all-loops -flto -mavx -mavx2 -mbmi -mpclmul -pedantic -Wall -Wextra -mvpclmulqdq
CFLAGS_VPCLMUL_GFNI:=-O3 -std=c99 -funroll-all-loops -flto -mavx -mavx2 -mbmi -mpclmul -pedantic -Wall -Wextra -mvpclmulqdq -mgfni

SHA3_SRC:=lib/fips202/fips202.c
SHA3_INCLUDE:=-I lib/fips202

SRC:=src/
INCLUDE:=-I src
LIB:=$(SHA3_INCLUDE)

MAIN_HQC:=src/main_hqc.c
MAIN_KAT:=src/main_kat.c

HQC_OBJS:=vector.o reed_muller.o reed_solomon.o fft.o gf.o gf2x.o code.o parsing.o hqc.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST_pclmul:=vector.o reed_muller.o reed_solomon.o fft.o gf.o gf2x_pclmul.o code.o parsing.o hqc_speed.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST_vpclmul:=vector.o reed_muller.o reed_solomon.o fft.o gf.o gf2x_vpclmul.o code.o parsing.o hqc_speed.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST_gfopt_pclmul:=vector.o reed_muller.o reed_solomon.o fft.o gf.o gf2x_gfopt_pclmul.o code.o parsing.o hqc_speed.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST_gfopt_vpclmul:=vector.o reed_muller.o reed_solomon.o fft.o gf.o gf2x_gfopt_vpclmul.o code.o parsing.o hqc_speed.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST_gf2xlib:=vector.o reed_muller.o reed_solomon.o fft.o gf.o gf2x_gf2xlib.o code.o parsing.o hqc_speed.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST_AFFT_COMMON:=vector.o reed_muller.o reed_solomon.o fft.o gf.o code.o parsing.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST_AFFT_MUL:=AFFT/bc_1_avx2.o AFFT/bc_256_avx2.o AFFT/btfy_avx2.o AFFT/btfy_consts_2stages.o AFFT/cachekey.o AFFT/cantor_to_gf264.o AFFT/cantor_to_gf264_2x.o AFFT/dencoder_avx2.o AFFT/gf2x.o AFFT/gf264_aesni.o AFFT/hqc_speed.o AFFT/polymul.o AFFT/polymul_hqclen_avx2pclmul.o
HQC_OBJS_SPEEDTEST_AFFT_MUL_GFNI:=AFFT/gfni/bc_1_avx2gfni.o AFFT/gfni/bc_256_avx2.o AFFT/gfni/btfy_gfni.o AFFT/gfni/cidx_to_gf256x2.o AFFT/btfy_consts_2stages.o AFFT/cachekey.o AFFT/cantor_to_gf264.o AFFT/cantor_to_gf264_2x.o AFFT/gfni/dencoder_gf256t4_avx2gfni.o AFFT/gf2x.o AFFT/gfni/gf256_gfni.o AFFT/hqc_speed.o AFFT/gfni/polymul.o AFFT/gfni/polymul_hqclen_avx2pclmul.o

HQC_OBJS_VERBOSE:=vector.o reed_muller.o reed_solomon-verbose.o fft.o gf.o gf2x.o code-verbose.o parsing.o hqc-verbose.o kem-verbose.o symmetric.o crypto_memset.o
LIB_OBJS:= fips202.o

BIN:=bin
BUILD:=bin/build

folders:
	@/bin/echo -e "\n### Creating folders"
	mkdir -p $(BUILD)

folders_AFFT:
	@/bin/echo -e "\n### Creating folders for AFFT"
	mkdir -p $(BUILD)/AFFT

folders_AFFT_gfni:
	@/bin/echo -e "\n### Creating folders for AFFT"
	mkdir -p $(BUILD)/AFFT/gfni

fips202.o: | folders
	@/bin/echo -e "\n### Compiling fips202"
	$(CC) $(CFLAGS) -c $(SHA3_SRC) $(SHA3_INCLUDE) -o $(BUILD)/$@

gf2x_pclmul.o: $(SRC)/gf2x.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -DGFMUL_VER=1 -o $(BUILD)/$@

gf2x_vpclmul.o: $(SRC)/gf2x.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS_VPCLMUL) -c $< $(INCLUDE) $(LIB) -DGFMUL_VER=2 -o $(BUILD)/$@

gf2x_gfopt_vpclmul.o: $(SRC)/gf2x_gfopt_vpclmul.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS_VPCLMUL) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%.o: $(SRC)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

AFFT/%.o: $(SRC)AFFT/%.c | folders folders_AFFT
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS_VPCLMUL) -c $< $(INCLUDE) $(INCLUDE_AFFT) $(LIB) -o $(BUILD)/$@
AFFT/gfni/%.o: $(SRC)AFFT/gfni/%.c | folders folders_AFFT folders_AFFT_gfni
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS_VPCLMUL_GFNI) -c $< $(INCLUDE) $(INCLUDE_AFFT) $(LIB) -o $(BUILD)/$@

%-verbose.o: $(SRC)/%.c | folders
	@/bin/echo -e "\n### Compiling $@ (verbose mode)"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -D VERBOSE -o $(BUILD)/$@

hqc-1: $(HQC_OBJS) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling hqc-1"
	$(CC) $(CFLAGS) $(MAIN_HQC) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

hqc-1-kat: $(HQC_OBJS) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling hqc-1 KAT"
	$(CC) $(CFLAGS) $(MAIN_KAT) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

hqc-1-verbose: $(HQC_OBJS_VERBOSE) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling hqc-1 (verbose mode)"
	$(CC) $(CFLAGS) $(MAIN_HQC) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -D VERBOSE -o $(BIN)/$@

test_speed_all: test/test_speed_pclmul test/test_speed_vpclmul test/test_speed_gfopt_pclmul test/test_speed_gfopt_vpclmul test/test_speed_gf2xlib

run_tests:
	@echo "\nRunning test/test_speed_pclmul..."
	@./test/test_speed_pclmul
	@echo "\nRunning test/test_speed_vpclmul..."
	@./test/test_speed_vpclmul
	@echo "\nRunning test/test_speed_gfopt_pclmul..."
	@./test/test_speed_gfopt_pclmul
	@echo "\nRunning test/test_speed_gfopt_vpclmul..."
	@./test/test_speed_gfopt_vpclmul
	@echo "\nRunning test/test_speed_gf2xlib..."
	@./test/test_speed_gf2xlib

test/test_speed_pclmul: $(HQC_OBJS_SPEEDTEST_pclmul) $(LIB_OBJS) | folders
	$(CC) $(CFLAGS) test/speed_print.c test/test_speed.c test/cpucycles.c -DGFMUL_VER=1 $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $@

test/test_speed_vpclmul: $(HQC_OBJS_SPEEDTEST_vpclmul) $(LIB_OBJS) | folders
	$(CC) $(CFLAGS_VPCLMUL) test/speed_print.c test/test_speed.c test/cpucycles.c -DGFMUL_VER=2 $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $@

test/test_speed_gfopt_pclmul: $(HQC_OBJS_SPEEDTEST_gfopt_pclmul) $(LIB_OBJS) | folders
	$(CC) $(CFLAGS) test/speed_print.c test/test_speed.c test/cpucycles.c -DGFMUL_VER=3 $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $@

test/test_speed_gfopt_vpclmul: $(HQC_OBJS_SPEEDTEST_gfopt_vpclmul) $(LIB_OBJS) | folders
	$(CC) $(CFLAGS_VPCLMUL) test/speed_print.c test/test_speed.c test/cpucycles.c -DGFMUL_VER=4 $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $@

test/test_speed_gf2xlib: $(HQC_OBJS_SPEEDTEST_gf2xlib) $(LIB_OBJS) | folders
	$(CC) $(CFLAGS) test/speed_print.c test/test_speed.c test/cpucycles.c -DGFMUL_VER=5 $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -l gf2x -o $@

test/test_speed_AFFT: $(HQC_OBJS_SPEEDTEST_AFFT_COMMON) $(HQC_OBJS_SPEEDTEST_AFFT_MUL) $(LIB_OBJS) | folders_AFFT
	$(CC) $(CFLAGS) test/speed_print.c test/test_speed.c test/cpucycles.c -DGFMUL_VER=6 $(addprefix $(BUILD)/, $^) $(INCLUDE) $(INCLUDE_AFFT) $(LIB) -D_GNU_SOURCE -o $@

test/test_speed_AFFT_gfni: $(HQC_OBJS_SPEEDTEST_AFFT_COMMON) $(HQC_OBJS_SPEEDTEST_AFFT_MUL_GFNI) $(LIB_OBJS) | folders_AFFT_gfni
	$(CC) $(CFLAGS_VPCLMUL_GFNI) test/speed_print.c test/test_speed.c test/cpucycles.c -DGFMUL_VER=6 $(addprefix $(BUILD)/, $^) $(INCLUDE) $(INCLUDE_AFFT) $(LIB) -D_GNU_SOURCE -o $@

clean:
	rm -f PQCkemKAT_*
	rm -f vgcore.*
	rm -rf ./bin

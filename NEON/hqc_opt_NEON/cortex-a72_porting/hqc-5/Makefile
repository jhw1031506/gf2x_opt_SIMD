CC:=gcc

CFLAGS:=-O3 -std=c99 -funroll-all-loops -flto -pedantic -Wall -Wextra -flax-vector-conversions

SHA3_SRC:=lib/fips202/fips202.c
SHA3_INCLUDE:=-I lib/fips202

SRC:=src/
INCLUDE:=-I src
INCLUDE_AFFT:=-I src/AFFT
LIB:=$(SHA3_INCLUDE)

MAIN_HQC:=src/main_hqc.c
MAIN_KAT:=src/main_kat.c

HQC_OBJS:=vector.o reed_muller.o reed_solomon.o fft.o gf.o gf2x.o code.o parsing.o hqc.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST:=vector.o reed_muller.o reed_solomon.o fft.o gf.o code.o parsing.o hqc_speed.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST_AFFT_COMMON:=vector.o reed_muller.o reed_solomon.o fft.o gf.o code.o parsing.o kem.o symmetric.o crypto_memset.o
HQC_OBJS_SPEEDTEST_AFFT_MUL:= AFFT/bc_8_neon.o AFFT/bc_128_neon.o AFFT/bitpoly_to_gf256x2_neon.o AFFT/btfy_neon.o AFFT/cachekey.o AFFT/combine.o AFFT/gf2x.o AFFT/gf256_neon.o AFFT/hqc_speed.o AFFT/polymul.o AFFT/polymul_crt_neon.o AFFT/ringmul.o AFFT/ringmul_neon.o
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

fips202.o: | folders
	@/bin/echo -e "\n### Compiling fips202"
	$(CC) $(CFLAGS) -c $(SHA3_SRC) $(SHA3_INCLUDE) -o $(BUILD)/$@

%.o: $(SRC)/%.c | folders
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

AFFT/%.o: $(SRC)AFFT/%.c | folders folders_AFFT
	@/bin/echo -e "\n### Compiling $@"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(INCLUDE_AFFT) $(LIB) -o $(BUILD)/$@

%-verbose.o: $(SRC)/%.c | folders
	@/bin/echo -e "\n### Compiling $@ (verbose mode)"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -D VERBOSE -o $(BUILD)/$@

hqc-5: $(HQC_OBJS) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling hqc-5"
	$(CC) $(CFLAGS) $(MAIN_HQC) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

hqc-5-kat: $(HQC_OBJS) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling hqc-5 KAT"
	$(CC) $(CFLAGS) $(MAIN_KAT) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

hqc-5-verbose: $(HQC_OBJS_VERBOSE) $(LIB_OBJS) | folders
	@/bin/echo -e "\n### Compiling hqc-5 (verbose mode)"
	$(CC) $(CFLAGS) $(MAIN_HQC) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -D VERBOSE -o $(BIN)/$@

test_speed_all: test/test_speed test/test_speed_gf2x_gfopt test/test_speed_AFFT

run_tests:
	@echo "\nRunning test/test_speed..."
	@./test/test_speed
	@echo "\nRunning test/test_speed_gf2x_gfopt..."
	@./test/test_speed_gf2x_gfopt
	@echo "\nRunning test/test_speed_AFFT..."
	@./test/test_speed_AFFT

test/test_speed: $(HQC_OBJS_SPEEDTEST) $(LIB_OBJS) | folders
	$(CC) $(CFLAGS) test/speed_print.c test/test_speed.c src/gf2x.c test/cpucycles.c $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -D_GNU_SOURCE -o $@

test/test_speed_gf2x_gfopt: $(HQC_OBJS_SPEEDTEST) $(LIB_OBJS) | folders
	$(CC) $(CFLAGS) test/speed_print.c test/test_speed.c src/gf2x_gfopt.c test/cpucycles.c $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -D_GNU_SOURCE -o $@

test/test_speed_AFFT: $(HQC_OBJS_SPEEDTEST_AFFT_COMMON) $(HQC_OBJS_SPEEDTEST_AFFT_MUL) $(LIB_OBJS) | folders_AFFT
	$(CC) $(CFLAGS_AFFT) test/speed_print.c test/test_speed.c test/cpucycles.c $(addprefix $(BUILD)/, $^) $(INCLUDE) $(INCLUDE_AFFT) $(LIB) -D_GNU_SOURCE -o $@

               
clean:
	rm -f PQCkemKAT_*
	rm -f vgcore.*
	rm -rf ./bin

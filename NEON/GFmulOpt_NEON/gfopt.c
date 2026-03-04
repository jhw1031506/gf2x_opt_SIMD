#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "gfopt.h"

int main()
{   
    FILE* fpr_0, * fpw0;
    
    //head_init -> head
    fpr_0 = fopen("./gfopt_head_init", "rb");
    fpw0 = fopen("./gfopt_head", "wb");
    char ff;
	while (!feof(fpr_0))
	{
		if(fread(&ff, 1, 1, fpr_0)!=1) break;
		fwrite(&ff, 1, 1, fpw0);
	}
    fclose(fpr_0);
    fclose(fpw0);

    char str[30]="./src/gfopt_basemul.c";

    FILE* fpr_1 = fopen("./gfopt_head", "rb");
    FILE* fpr_3 = fopen("./gfopt_tail_basemul", "rb");
	FILE* fpw = fopen(str, "ab");
    char file;

    fprintf(fpw,"#define GFLEN 1\n");

	while (!feof(fpr_1))
	{
		if(fread(&file, 1, 1, fpr_1)!=1) break;
		fwrite(&file, 1, 1, fpw);
	}
    fclose(fpr_1);
	while (!feof(fpr_3))
	{
		if(fread(&file, 1, 1, fpr_3)!=1) break;
		fwrite(&file, 1, 1, fpw);
	}
    fclose(fpr_3);
	fclose(fpw);
    
    char strsys1[200], strsys2[200];
    sprintf(strsys1,"gcc  ./src/gfopt_basemul.c ./tools.c -o ./src/gfopt_basemul -O3 -funroll-all-loops  -pedantic -Wall -Wextra -DENV_NEON");
    if(system(strsys1)) printf("./src/gfopt_basemul.c compile err\n");
    clock_t sleep_t=clock();
    while(1){
        if((clock()-sleep_t)/CLOCKS_PER_SEC >= 1) break;
    }
    sprintf(strsys2,"./src/gfopt_basemul");
    if(system(strsys2)) printf("./src/gfopt_basemul err\n");
    
    return 0;

}
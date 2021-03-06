
#include <stdio.h>
#include <stdlib.h>
#include "generate.h"
#include "tensor_handling.hpp"

#define SHAPE 16
#define ITER 1

int main(){
    float variance = 0.01;
    int n_mols = 0;
    FILE* f = fopen("test_input.mol", "r");
    fread(&n_mols, 4, 1, f);
    int* sizes = (int*)malloc(n_mols*sizeof(int));
    float** mols = (float**)malloc(n_mols*sizeof(float*));
    float* tensor = (float*)aligned_alloc(512, n_mols*8*SHAPE*SHAPE*SHAPE*sizeof(float));
    if(!tensor){
        printf("Failed to allocate size for tensor: aborting\n");
        exit(1);
    }
    for(int i = 0; i < n_mols; i++){
        fread(&sizes[i], 4, 1, f);
        mols[i] = (float*)malloc(sizes[i]*4*sizeof(float));
        fread(mols[i], sizeof(float), sizes[i]*4, f);
    }
    fclose(f);
    int N = 8;
    for(int j = 0; j < ITER; j++){
        for(int i = 0; i < n_mols; i++){
            float* points = mols[i];
            OutputSpec o;
            ssize_t shape[] = {8, SHAPE, SHAPE, SHAPE};
            fill_output_spec(&o, 
                    sizes[i], points,
                    4, shape, nullptr,
                    variance,
                    nullptr,
                    nullptr);
            gaussian_erf(sizes[i], points, &o, tensor+i*N*SHAPE*SHAPE*SHAPE);
        }
    }
    for(int i = 0; i < n_mols; i++){
        free(mols[i]);
    }
    free(sizes);
    free(mols);
    display_tensor_xy(SHAPE, SHAPE, SHAPE, tensor, 4);
    free(tensor);
}

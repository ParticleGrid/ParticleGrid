
#include <stdio.h>
#include <stdlib.h>
#include "generate.h"
#include "tensor_handling.hpp"

#define SHAPE 256

int main(){
    float variance = 0.1;
    float ic = (float)(1/(SQRT_2*variance));
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
    for(int j = 0; j < 10; j++){
        for(int i = 0; i < n_mols; i++){
            OutputSpec o;
            fill_output_spec(&o, variance, SHAPE, SHAPE, SHAPE, N, nullptr);
            get_grid_extent(sizes[i], points, o.ext);
            float* points = mols[i];
            gaussian_erf(sizes[i], points, &o, tensor+i*8*SHAPE*SHAPE*SHAPE);
        }
    }
}

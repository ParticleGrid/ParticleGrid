
#include <stdio.h>
#include "generate.h"
#include "tensor_handling.hpp"

#define SHAPE 64

int main(){
    float variance = 0.1;
    float ic = (float)(1/(SQRT_2*variance));
    int n_mols = 0;
    FILE* f = fopen("test_input.mol", "r");
    fread(&n_mols, 4, 1, f);
    printf("%d\n", n_mols);
    int* sizes = (int*)malloc(n_mols*sizeof(int));
    float** mols = (float**)malloc(n_mols*sizeof(float*));
    float* tensor = (float*)malloc(n_mols*8*SHAPE*SHAPE*SHAPE*sizeof(float));
    for(int i = 0; i < n_mols; i++){
        fread(&sizes[i], 4, 1, f);
        mols[i] = (float*)malloc(sizes[i]*4*sizeof(float));
        fread(mols[i], sizeof(float), sizes[i]*4, f);
    }
    for(int i = 0; i < n_mols; i++){
        OutputSpec o = {
            .ext = {},
            .erf_inner_c = ic,
            .erf_outer_c = 0.125,
            .W = SHAPE,
            .H = SHAPE,
            .D = SHAPE,
            .N = 8
        };
        float* points = mols[i];
        get_grid_extent(sizes[i], points, o.ext);
        gaussian_erf(sizes[i], points, &o, tensor+i*8*SHAPE*SHAPE*SHAPE);
        if(i==1){
            display_tensor_xy(SHAPE, SHAPE, SHAPE, tensor, 4);
        }
    }
}

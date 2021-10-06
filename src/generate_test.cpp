
#include <stdio.h>
#include "generate.h"
#include "tensor_handling.hpp"

#define ERFN 16

int main(){
    float pos = 0.5;
    float grid_dim = 1;
    float cell_dim = grid_dim/ERFN;
    float variance = 0.04;
    float ic = (float)(1/(SQRT_2*variance));
    alignas (32) float erfa[ERFN+1];
    float erfb[ERFN+1];
    #ifdef V8F
    /*
    v8sf delta;
    for(int idx = 0; idx < 8; idx++){
        delta[idx] = idx*cell_dim-pos;
    }
    // calculate erf at every cell
    erf_helper_v8f(erfa, delta, ERFN, ic, cell_dim, grid_dim);
    erf_helper(erfb, pos, ERFN, ic, cell_dim, grid_dim);
    for(int i = 0; i < ERFN; i++){
        float d = erfa[i] - erfb[i];
        if(d != 0){
            printf("%e ", d);
        }
    }
    printf("\n");
    */
    #endif
    float* tensor = (float*)malloc(8*ERFN*ERFN*ERFN*sizeof(float));
    float points[] = {
        0.0, 0.2, 0.3, 0.5,
    };
    OutputSpec o = {
        .ext = {
            {0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0}
        },
        .erf_inner_c = ic,
        .erf_outer_c = 0.125,
        .W = ERFN,
        .H = ERFN,
        .D = ERFN,
        .N = 8
    };
    gaussian_erf(1, points, &o, tensor);
    display_tensor_xy(ERFN, ERFN, ERFN, tensor, 4);
}

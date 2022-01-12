#pragma once

#include "perfdef.h"
#include "avx_erf.hpp"

#define SQRT_2 1.41421356237
#define INTEGRAL_NORMALIZATION_3D 0.125

typedef struct OutputSpec {
    float ext[2][3];
    float erf_inner_c[3];
    float erf_outer_c;
    int shape[3];
    int N;
} OutputSpec;

#ifdef V8F
static inline void erf_helper_v8f(float* erf_i, v8sf delta, int bound, float ic, float cell_i, float dim_size){
    // Calculate error function at each grid points

    for (int i = 0; i+8 <= bound; i+=8){
        v8sf erfv = erf256_ps(delta * ic);
        _mm256_store_ps(erf_i+i, erfv);
        delta += cell_i * 8;
    }
    erf_i[bound] = erf_apx(dim_size*ic);

    // Calculate error function difference at each interval in the grid point
    for(int i = 0; i+8 <= bound; i+=8){
        v8sf erfv = _mm256_load_ps(erf_i+i);
        v8sf erfv2 = _mm256_loadu_ps(erf_i+i+1);
        _mm256_store_ps(erf_i+i, erfv2-erfv);
    }
}
#endif

static inline void erf_helper(float* erf_i, float pos, int bound, float ic, float cell_i, float dim_size){
    // Calculate error function at each grid points

    for (int i = 0; i <= bound; i++){
        float dist = i * cell_i - pos;
        float erfv = erf_apx(dist * ic);
        erf_i[i] = erfv;
    }

    // Calculate error function difference at each interval in the grid point
    for(int i = 0; i < bound; i++){
        float v = erf_i[i+1] - erf_i[i];
        erf_i[i] = v;
    }
}

static inline bool erf_range_helper(int range[2], int center, int grid_shape, float* erf_i){
    // calculate the range where the erf is non-zero (inclusive)
    // return true if the range can be skipped (is all zeros)
    int i = center;
    if(i < 0) i = 0;
    if(i > grid_shape) i = grid_shape;
    center = i;
    if(erf_i[i] == 0.0){
        return 1;
    }
    while(i < grid_shape && erf_i[i++] != 0.0);
    i--;
    range[1] = i;
    i = center;
    while(i >= 0 && erf_i[i--] != 0.0);
    i++;
    range[0] = i;
    return 0;
}

// The following is not a true header file. it is a function
// It is created twice with different names by these includes
#ifdef V8F
#define GAUSS_V8F
#endif
#include "gaussian_erf.h"

#undef GAUSS_V8F
#define ERF_FN_NAME gaussian_erf_noavx
#include "gaussian_erf.h"

// data setup helpers
void get_grid_extent(size_t n_atoms, float* points, float ret_extent[2][3]){
    /**
     * creates an extent which is slightly larger than the bounding box of all provided points
     */
    for(int j = 0; j < 3; j++){
        ret_extent[0][j] = points[j+1];
        ret_extent[1][j] = points[j+1];
    }
    for(size_t i = 1; i < n_atoms; i++){
        for(int j = 0; j < 3; j++){
            float v = points[i*4 + j + 1];
            if(v < ret_extent[0][j])
                ret_extent[0][j] = v;
            if(v > ret_extent[1][j])
                ret_extent[1][j] = v;
        }
    }        
    for(int j = 0; j < 3; j++){
        ret_extent[0][j] -= 2;
    }
    for(int j = 0; j < 3; j++){
        ret_extent[1][j] += 2;
    }
}

void fill_output_spec(OutputSpec* ospec, float variance, int W, int H, int D, int N, float* extent) {
    *ospec = {
        .ext = {},
        .erf_inner_c = {},
        .erf_outer_c = 0.125,
        .shape = {W, H, D},
        .N = N
    };
    if(extent){
        memcpy(&ospec->ext, extent, 6*sizeof(float));
    }
    for(int i3 = 0; i3 < 3; i3++){
        ospec->erf_inner_c[i3] = (float)(ospec->shape[i3]/(ospec->ext[i3]*SQRT_2*variance))
    }
}

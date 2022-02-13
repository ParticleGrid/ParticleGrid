#pragma once

#include <string.h>
#include <assert.h>

#include "perfdef.h"
#include "avx_erf.hpp"

#define SQRT_2 1.41421356237
#define INTEGRAL_NORMALIZATION_3D 0.125

struct Options {
    bool dynamic_variance = true;
    bool collapse_channel = false;
    const float* channel_weights = nullptr;
};

typedef struct OutputSpec {
    float ext[2][3];
    float erf_inner_c[3];
    float erf_outer_c;
    size_t shape[4];
    size_t strides[4];
    bool collapse_channel;
    const float* channel_weights;
} OutputSpec;

struct atom_spec_t {
    size_t grid_shape[4];
    size_t grid_strides[4];
    float point[3];
    int channel;
    float weight;
    // dimensions for full grid, in arbitrary units
    float grid_dim[3];
    // dimensions for individual grid cells
    float cell_dim[3];
    float erf_inner_c[3];
    float erf_outer_c;
    float* tensor;
};

static inline size_t tensor_size(OutputSpec* o){
    return o->strides[4]*o->shape[4];
}

#ifdef V8F
static inline void erf_helper_v8f(float* erf_i, v8sf delta, int bound, float ic, float cell_i, float dim_size){
    // Calculate error function at each grid points. erf_i should be an array with 8 floats of padding at the end

    int i;
    for (i = 0; i <= bound; i+=8){
        v8sf erfv = erf256_ps(delta * ic);
        _mm256_store_ps(erf_i+i, erfv);
        delta += cell_i * 8;
    }

    // Calculate error function difference at each interval in the grid point
    for(int i = 0; i < bound; i+=8){
        v8sf erfv = _mm256_load_ps(erf_i+i);
        v8sf erfv2 = _mm256_loadu_ps(erf_i+i+1);
        v8sf res = erfv2-erfv;
        _mm256_store_ps(erf_i+i, res);
    }
    
    for(int i = bound; i < bound + 8; i++) {
        erf_i[i] = 0;
    }
    // printf("BOUND %d\n", bound);
    // for(int i = 0; i < bound; i++){
        // printf("%.3e ", erf_i[i]);
    // }
    // printf("\n");
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

static inline bool erf_range_helper(size_t range[2], int center, int grid_shape, float* erf_i){
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

// The following is not a true header file. It is a function.
// It is created twice with different names by these includes
#ifdef V8F
#define GAUSS_V8F
#endif
#include "gaussian_erf.h"

#undef GAUSS_V8F
#define GAUSS_SUFFIX _noavx
#include "gaussian_erf.h"

// data setup helpers
void get_grid_extent(size_t n_atoms, const float* points, float ret_extent[2][3]){
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
        ret_extent[1][j] += 2;
    }
}

void fill_output_spec(OutputSpec* ospec, 
        ssize_t num_points, const float* points, 
        ssize_t num_dim, const ssize_t* shape, const ssize_t* strides, 
        float variance, 
        const float* extent, 
        const Options* ptr_options) {
    Options options;
    if(ptr_options){
        options = *ptr_options;
    }

    assert(num_dim == 4); // N, D, H, W
    *ospec = {
        .ext = {},
        .erf_inner_c = {},
        .erf_outer_c = 0.125,
        .shape = {},
        .strides = {}, 
        .collapse_channel = options.collapse_channel,
        .channel_weights = options.channel_weights
    };
    if(extent) {
        memcpy((float*)&ospec->ext[0], extent, 6*sizeof(float));
    }
    else{
        get_grid_extent(num_points, points, ospec->ext);
    }
    memcpy(&ospec->shape[0], shape, 4*sizeof(ospec->shape[0]));
    if(strides){
        for(int i = 0; i < 4; i++){
            assert(strides[i] % sizeof(float) == 0);
            ospec->strides[i] = strides[i]/sizeof(float);
        }
    }
    else{
        ospec->strides[3] = 1;
        for(int i = 2; i >= 0; i--){
            ospec->strides[i] = ospec->strides[i+1] * shape[i+1];
        }
    }
    // printf("B %ld %ld %ld %ld\n", ospec->strides[0], ospec->strides[1], ospec->strides[2], ospec->strides[3]);
    // printf("C %ld %ld %ld %ld\n", ospec->shape[0], ospec->shape[1], ospec->shape[2], ospec->shape[3]);
    for(int i = 0; i < 3; i++){
        // float span = ospec->ext[1][i] - ospec->ext[0][i];
        float ic = (float)(512/(SQRT_2*ospec->shape[3-i]));
        if(options.dynamic_variance){
            ic = ic / variance;
        }
        ospec->erf_inner_c[i] = ic;
    }
}

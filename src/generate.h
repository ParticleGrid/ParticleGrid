#pragma once

#include "perfdef.h"
#include "avx_erf.hpp"

#define SQRT_2 1.41421356237
#define INTEGRAL_NORMALIZATION_3D 0.125

typedef struct OutputSpec {
    float ext[2][3];
    float erf_inner_c, erf_outer_c;
    int W, H, D;
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

static inline void gaussian_erf(size_t n_atoms, const float* points, OutputSpec* o, float* tensor){
    /**
     * Generate a tensor in `tensor` using the information provided
     *
     * `points` is an array of size `n_atoms*4`
     * Where data comes in the form {c, x, y, z}
     *      c is the channel (as a float), (x, y, z) is the atom's position
     */
    int grid_shape[3] = {
        o->W,
        o->H,
        o->D
    };
    float ic = o->erf_inner_c;
    float oc = o->erf_outer_c;
    // dimensions for full grid
    float grid_dim[3];
    // dimensions for individual grid cells
    float cell_dim[3];
    for(int i = 0; i < 3; i++){
        grid_dim[i] = o->ext[1][i] - o->ext[0][i];
        cell_dim[i] = grid_dim[i]/grid_shape[i];
    };
    // where data will be stored about the erf function calculations
    alignas (32) float erfx[o->W+1];
    alignas (32) float erfy[o->H+1];
    alignas (32) float erfz[o->D+1];
    float* erfs[] = {
        erfx, erfy, erfz
    };
    
    for(size_t point = 0; point < n_atoms; point++){
        int channel = (int)points[point*4];
        int tens_offset = o->W*o->H*o->D*channel;
        float pos[3];
        int atom_spans[3][2];
        #ifdef V8F
        v8sf posv[3];
        #endif
        // get atom positions
        for(int i = 0; i < 3; i++){
            pos[i] = points[point*4 + i + 1] - o->ext[0][i];
            #ifdef V8F
            posv[i] = _mm256_broadcast_ss(&pos[i]);
            #endif
        }

        // represents spaces between cells
        #ifdef V8F
        v8sf offsets[3];
        #endif
        bool skip = 0;
        for(int dim = 0; dim < 3; dim++){
            #ifdef V8F
            for(int idx = 0; idx < 8; idx++){
                offsets[dim][idx] = idx*cell_dim[dim];
            }
            // calculate erf at every cell
            v8sf delta = offsets[dim] - posv[dim];
            erf_helper_v8f(erfs[dim], delta, grid_shape[dim], ic, cell_dim[dim], grid_dim[dim]);
            #else
            erf_helper(erfs[dim], pos[dim], grid_shape[dim], ic, cell_dim[dim], grid_dim[dim]);
            #endif
            #ifndef NO_SPARSE
            int center = (int)( (pos[dim])/cell_dim[dim] );
            skip = erf_range_helper(atom_spans[dim], center, grid_shape[dim], erfs[dim]);
            if(skip) break;
            #else
            atom_spans[dim][0] = 0;
            atom_spans[dim][1] = grid_shape[dim]-1;
            #endif
        }
        if(skip){
            continue;
        }

        #ifdef V8F
        if(atom_spans[0][0] % 8)
            atom_spans[0][0] -= atom_spans[0][0] % 8;
        if(atom_spans[0][1] % 8 == 0)
            atom_spans[0][1] += 8;
        #endif

        int HH = o->H;
        int WW = o->W;

        // multiply the erf values together to get the final volume integration
        #ifdef OMP_ON
        #pragma omp parallel for
        #endif
        for(int k = atom_spans[2][0]; k <= atom_spans[2][1]; k++){
            float z_erf = erfz[k] * oc;
            size_t koff = k*WW*HH + tens_offset;
            for(int j = atom_spans[1][0]; j <= atom_spans[1][1]; j++){
                float yz_erf = erfy[j] * z_erf;
                size_t kjoff = koff + j*WW;
                for(int i = atom_spans[0][0]; i < atom_spans[0][1]; i+=VSIZE){
                    size_t idx = i + kjoff;
                    #ifdef V8F
                    v8sf erfxv = _mm256_load_ps(erfx+i);
                    v8sf tmp = _mm256_loadu_ps(&tensor[idx]);
                    tmp = tmp + (erfxv * yz_erf);
                    _mm256_storeu_ps(&tensor[idx], tmp);
                    #else
                    float v = erfx[i]*yz_erf;
                    tensor[idx] += v;
                    #endif
                }
            }
        }
    }
}

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


#pragma once

#include "avx_erf.hpp"

#define SQRT_2 1.41421356237
#define INTEGRAL_NORMALIZATION_3D 0.125

typedef struct OutputSpec {
    float ext[2][3];
    float erf_inner_c, erf_outer_c;
    int W, H, D;
    int N;
} OutputSpec;

static inline void simd_err_func_helper(float* erf_i, v8sf delta, int bound, float ic, float cell_i, float dim_size){
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

static inline void gaussian_erf_avx_sparse(size_t n_atoms, const float* points, OutputSpec* o, float* tensor){
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
        v8sf posv[3];
        // get atom positions
        for(int i = 0; i < 3; i++){
            pos[i] = points[point*4 + i + 1] - o->ext[0][i];
            posv[i] = _mm256_broadcast_ss(&pos[i]);
        }

        // represents spaces between cells
        v8sf offsets[3];
        bool skip = 0;
        for(int dim = 0; dim < 3; dim++){
            memset(erfs[dim], 0, (grid_dim[dim] + 1) * sizeof(float));
            int center = (int)( (pos[dim])/cell_dim[dim] );
            for(int idx = 0; idx < 8; idx++){
                offsets[dim][idx] = idx*cell_dim[dim];
            }
            // calculate erf at every cell
            v8sf delta = offsets[dim] - posv[dim];
            simd_err_func_helper(erfs[dim], delta, grid_shape[dim], ic, cell_dim[dim], grid_dim[dim]);
            int i = center;
            if(i < 0) i = 0;
            if(i > grid_shape[dim]) i = grid_shape[dim];
            center = i;
            if(erfs[dim][i] == 0.0){
                skip = true;
                break;
            }
            while(erfs[dim][i] != 0.0 && i < grid_shape[dim]+1){
                i++;
            }
            i--;
            atom_spans[dim][1] = i;
            i = center;
            while(erfs[dim][i] != 0.0 && i > 0){
                i--;
            }
            i++;
            atom_spans[dim][0] = i;
        }
        if(skip){
            continue;
        }
        // avx 256-bit vector specific
        if(atom_spans[0][0] % 8)
            atom_spans[0][0] -= atom_spans[0][0] % 8;
        if(atom_spans[0][1] % 8 == 0)
            atom_spans[0][1] += 8;

        int HH = o->H;
        int WW = o->W;

        // multiply the erf values together to get the final volume integration
        for(int k = atom_spans[2][0]; k <= atom_spans[2][1]; k++){
            float z_erf = erfz[k] * oc;
            size_t koff = k*WW*HH + tens_offset;
            for(int j = atom_spans[1][0]; j <= atom_spans[1][1]; j++){
                float yz_erf = erfy[j] * z_erf;
                size_t kjoff = koff + j*WW;
                for(int i = atom_spans[0][0]; i < atom_spans[0][1]; i+=8){
                    v8sf erfxv = _mm256_load_ps(erfx+i);
                    size_t idx = i + kjoff;
                    v8sf tmp = _mm256_loadu_ps(&tensor[idx]);
                    tmp = tmp + (erfxv * yz_erf);
                    _mm256_storeu_ps(&tensor[idx], tmp);
                }
            }
        }
    }
}


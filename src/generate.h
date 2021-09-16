
#include "molecule.hpp"
#include "avx_erf.hpp"

#define SQRT_2 1.41421356237
#define INTEGRAL_NORMALIZATION_3D 0.125

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

static inline void multithreaded_gaussian_erf_avx(size_t n_atoms, const float* points, GridSpec* o, float* tensor){
    const float width = o->width;
    const float height = o->height;
    const float depth = o->depth;
    const float variance = o->variance;
    const float cell_w = width/o->grid_size;
    const float cell_h = height/o->grid_size;
    const float cell_d = depth/o->grid_size;
    const float ic = (float)(1/(SQRT_2 * variance));
    const float oc = INTEGRAL_NORMALIZATION_3D;


    #pragma omp parallel for
    for(size_t point = 0; point < n_atoms; ++point){

        // Do the error function calculations in parallel

        alignas (32) float erfx[o->grid_size+1];
        alignas (32) float erfy[o->grid_size+1];
        alignas (32) float erfz[o->grid_size+1];
    int channel = (int)points[point*4];
        float x = points[point*4 + 1];
        float y = points[point*4 + 2];
        float z = points[point*4 + 3];
        int tens_offset = o->grid_size*o->grid_size*o->grid_size*channel;
        v8sf x_atom, y_atom, z_atom;
        x_atom = _mm256_broadcast_ss(&x);
        y_atom = _mm256_broadcast_ss(&y);
        z_atom = _mm256_broadcast_ss(&z);

        v8sf x_offset, y_offset, z_offset;
        for(int idx = 0; idx < 8; idx++){
            x_offset[idx] = idx*cell_w;
            y_offset[idx] = idx*cell_h;
            z_offset[idx] = idx*cell_d;
        }

        v8sf delta = x_offset - x_atom;
        simd_err_func_helper(erfx, delta, o->grid_size, ic, cell_w, width);

        delta = y_offset - y_atom;
        simd_err_func_helper(erfy, delta, o->grid_size, ic, cell_h, height);
        
        delta = z_offset - z_atom;
        simd_err_func_helper(erfz, delta, o->grid_size, ic, cell_d, depth);

        // Update tensor sequentially
        #pragma omp critical
        {
            int DD = o->grid_size;
            int HH = o->grid_size;
            int WW = o->grid_size;

            for(int k = 0; k < DD; k++){
                float z_erf = erfz[k] * oc;
                size_t koff = k*WW*HH + tens_offset;
                for(int j = 0; j < HH; j++){
                    float yz_erf = erfy[j] * z_erf;
                    #ifdef FORCE_FMA
                    v8sf yz_erfv = _mm256_broadcast_ss(&yz_erf);
                    #endif
                    size_t kjoff = koff + j*WW;
                    for(int i = 0; i+8 <= WW; i+=8){
                        v8sf erfxv = _mm256_load_ps(erfx+i);
                        size_t idx = i + kjoff;
                        v8sf tmp = _mm256_loadu_ps(&tensor[idx]);
                        #ifdef FORCE_FMA
                        tmp = _mm256_fmadd_ps(erfxv, yz_erfv, tmp);
                        #else
                        tmp = tmp + (erfxv * yz_erf);
                        #endif
                        _mm256_storeu_ps(&tensor[idx], tmp);
                    }
                }
            }
        }
    }
}

static inline void gaussian_erf_avx_fit_points(size_t n_atoms, float* points, OutputSpec* o, float* tensor){
    /**
     * Generate a tensor in `tensor` using the information provided
     *
     * `points` is an array of size `n_atoms*4`
     * Where data comes in the form {c, x, y, z}
     *      c is the channel (as a float), (x, y, z) is the atom's position
     */
    float width = o->ext[1][0] - o->ext[0][0];
    float height = o->ext[1][1] - o->ext[0][1];
    float depth = o->ext[1][2] - o->ext[0][2];

    // dimensions for individual grid cells
    float cell_w = width/o->W;
    float cell_h = height/o->H;
    float cell_d = depth/o->D;
    // info for calculating the erf function
    float ic = o->erf_inner_c;
    float oc = o->erf_outer_c;
    // where data will be stored about the erf function calculations
    alignas (32) float erfx[o->W+1];
    alignas (32) float erfy[o->H+1];
    alignas (32) float erfz[o->D+1];
    
    for(size_t point = 0; point < n_atoms; point++){
        int channel = (int)points[point*4];
        float x = points[point*4 + 1] - o->ext[0][0];
        float y = points[point*4 + 2] - o->ext[0][1];
        float z = points[point*4 + 3] - o->ext[0][2];
        int tens_offset = o->W*o->H*o->D*channel;
        v8sf x_atom, y_atom, z_atom;
        x_atom = _mm256_broadcast_ss(&x);
        y_atom = _mm256_broadcast_ss(&y);
        z_atom = _mm256_broadcast_ss(&z);

        // represents spaces between cells
        v8sf x_offset, y_offset, z_offset;
        for(int idx = 0; idx < 8; idx++){
            x_offset[idx] = idx*cell_w;
            y_offset[idx] = idx*cell_h;
            z_offset[idx] = idx*cell_d;
        }

        // calculate the erf function for every cell
        v8sf delta = x_offset - x_atom;
        simd_err_func_helper(erfx, delta, o->W, ic, cell_w, width);

        delta = y_offset - y_atom;
        simd_err_func_helper(erfy, delta, o->H, ic, cell_h, height);
        
        delta = z_offset - z_atom;
        simd_err_func_helper(erfz, delta, o->D, ic, cell_d, depth);

        int DD = o->D;
        int HH = o->H;
        int WW = o->W;

        // multiply the erf values together to get the final volume integration
        for(int k = 0; k < DD; k++){
            float z_erf = erfz[k] * oc;
            size_t koff = k*WW*HH + tens_offset;
            for(int j = 0; j < HH; j++){
                float yz_erf = erfy[j] * z_erf;
                #ifdef FORCE_FMA
                v8sf yz_erfv = _mm256_broadcast_ss(&yz_erf);
                #endif
                size_t kjoff = koff + j*WW;
                for(int i = 0; i+8 <= WW; i+=8){
                    v8sf erfxv = _mm256_load_ps(erfx+i);
                    size_t idx = i + kjoff;
                    v8sf tmp = _mm256_loadu_ps(&tensor[idx]);
                    #ifdef FORCE_FMA
                    tmp = _mm256_fmadd_ps(erfxv, yz_erfv, tmp);
                    #else
                    tmp = tmp + (erfxv * yz_erf);
                    #endif
                    _mm256_storeu_ps(&tensor[idx], tmp);
                }
            }
        }
    }
}

static inline void gaussian_erf_avx_sparse(size_t n_atoms, float* points, OutputSpec* o, float* tensor){
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


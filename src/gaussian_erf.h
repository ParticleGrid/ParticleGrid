
// not a regular header file, and *should not* have include guards

#define PASTER(x,y) x ## y
#define EVALUATOR(x,y)  PASTER(x,y)
#ifndef GAUSS_SUFFIX
#define _GAUSS_SUFFIX
#else
#define _GAUSS_SUFFIX EVALUATOR(_, GAUSS_SUFFIX)
#endif
#define GAUSS_FN EVALUATOR(gaussian_erf, _GAUSS_SUFFIX)
#define GAUSS_ATOM_FN EVALUATOR(gaussian_erf, _GAUSS_SUFFIX)

static inline void GAUSS_ATOM_FN(atom_spec_t* atom_spec){
    size_t* grid_shape = atom_spec->grid_shape;
    size_t* grid_strides = atom_spec->grid_strides;
    float* tens_offset = atom_spec->tensor;
    float* grid_dim = atom_spec->grid_dim;
    float* cell_dim = atom_spec->cell_dim;
    float* ic = atom_spec->erf_inner_c;
    float oc = atom_spec->erf_outer_c;
    size_t atom_spans[3][2];
    // printf("B %ld %ld %ld %ld\n", grid_strides[0], grid_strides[1], grid_strides[2], grid_strides[3]);
    // printf("C %ld %ld %ld %ld\n", grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3]);
    alignas (32) float erfx[grid_shape[4-1]+8];
    alignas (32) float erfy[grid_shape[4-2]+8];
    alignas (32) float erfz[grid_shape[4-3]+8];
    float* erfs[] = {
        erfx, erfy, erfz
    };
    float* pos = atom_spec->point;
    #ifdef GAUSS_V8F
    v8sf posv[3];
    // get atom positions as v8sf
    for(int i = 0; i < 3; i++){
        posv[i] = _mm256_broadcast_ss(&pos[i]);
    }
    #endif

    // represents spaces between cells
    #ifdef GAUSS_V8F
    v8sf offsets[3];
    #endif
    bool skip = 0;
    for(int dim = 0; dim < 3; dim++){
    #ifdef GAUSS_V8F
        for(int idx = 0; idx < 8; idx++){
            offsets[dim][idx] = idx*cell_dim[dim];
        }
        // calculate erf at every cell
        v8sf delta = offsets[dim] - posv[dim];
        erf_helper_v8f(erfs[dim], delta, grid_shape[4-1-dim], ic[dim], cell_dim[dim], grid_dim[dim]);
    #else
        erf_helper(erfs[dim], pos[dim], grid_shape[4-1-dim], ic[dim], cell_dim[dim], grid_dim[dim]);
    #endif
    #ifndef NO_SPARSE
        int center = (int)( (pos[dim])/cell_dim[dim] );
        skip = erf_range_helper(atom_spans[dim], center, grid_shape[4-1-dim], erfs[dim]);
        if(skip) break;
    #else
        atom_spans[dim][0] = 0;
        atom_spans[dim][1] = grid_shape[4-1-dim]-1;
    #endif
    }
    if(skip){
        return;
    }

    #ifdef GAUSS_V8F
    int miss = atom_spans[0][0] % 8;
    if(miss)
        atom_spans[0][0] -= miss;
    if(atom_spans[0][1] % 8 == 0)
        atom_spans[0][1] += 8;
    #endif

    // multiply the erf values together to get the final volume integration
    // #ifdef OMP_ON
    // #pragma omp parallel for
    // #endif
    for(size_t k = atom_spans[2][0]; k <= atom_spans[2][1]; k++){
        float z_erf = erfz[k] * oc * atom_spec->weight;
        float* koff = tens_offset + k*grid_strides[1];
        for(size_t j = atom_spans[1][0]; j <= atom_spans[1][1]; j++){
            float yz_erf = erfy[j] * z_erf;
            float* kjoff = koff + j*grid_strides[2];
            for(size_t i = atom_spans[0][0]; i < atom_spans[0][1]; i+=VSIZE){
                float* idx = kjoff + i; // TODO grid_strides[3] ? doesn't work with simd
                #ifdef GAUSS_V8F
                if(idx+8 >= tens_offset + grid_strides[0]){
                    // printf("AAA %ld\n", idx-tens_offset);
                    for(size_t i2 = i; i2 < atom_spans[0][1] && idx < tens_offset + grid_strides[0]; i2++, idx++){
                        *idx += erfx[i2]*yz_erf;
                    }
                    // printf("BBB %ld\n", idx+8-tens_offset);
                }
                else{
                    v8sf erfxv = _mm256_load_ps(erfx+i);
                    v8sf tmp = _mm256_loadu_ps(idx);
                    tmp = tmp + erfxv * yz_erf;
                    _mm256_storeu_ps(idx, tmp);
                }
                for(int i2 = 0; i2 < 8 && idx + i2 < tens_offset + grid_strides[0]; i2++){
                    if(idx[i2] > 1000 || idx[i2] < -0.00001){
                        // printf("BAD %ld %ld %ld %ld: %f\n", i2, i, j, k, idx[i2]);
                    }
                }
                // printf("BBB %ld\n", idx-tens_offset);
                #else
                float v = erfx[i]*yz_erf;
                *idx += v;
                #endif
            }
        }
    }
}

static inline void GAUSS_FN(size_t n_atoms, const float* points, OutputSpec* o, float* tensor){
    /**
     * Generate a tensor in `tensor` using the information provided
     *
     * `points` is an array of size `n_atoms*4`
     * Where data comes in the form {c, x, y, z}
     *      c is the channel (as a float), (x, y, z) is the atom's position
     */
    atom_spec_t atom_spec;
    for(int i = 0; i < 4; i++){
        atom_spec.grid_shape[i] = o->shape[i];
        atom_spec.grid_strides[i] = o->strides[i];
    };
    for(int i = 0; i < 3; i++){
        atom_spec.grid_dim[i] = o->ext[1][i] - o->ext[0][i];
        atom_spec.cell_dim[i] = atom_spec.grid_dim[i]/atom_spec.grid_shape[4-1-i];
        atom_spec.erf_inner_c[i] = o->erf_inner_c[i];
    };
    atom_spec.erf_outer_c = o->erf_outer_c;
    // where data will be stored about the erf function calculations
    
    for(size_t atom = 0; atom < n_atoms; atom++){
        size_t idx = atom*4;
        atom_spec.channel = (int)points[idx];
        for(int i = 0; i < 3; i++){
            atom_spec.point[i] = points[(idx + 1) + i] - o->ext[0][i];
        }
        atom_spec.tensor = tensor;
        if(!o->collapse_channel) {
            atom_spec.tensor += atom_spec.grid_strides[0]*atom_spec.channel;
        }
        if(o->channel_weights) {
            atom_spec.weight = o->channel_weights[atom_spec.channel];
        }
        else {
            atom_spec.weight = 1;
        }
        GAUSS_ATOM_FN(&atom_spec);
    }
}

#undef _GAUSS_SUFFIX

#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>

#include <omp.h>

#include "cgridgen.h"

#include "avx_erf.hpp"

#define SQRT_2 1.41421356237
#define INTEGRAL_NORMALIZATION_3D 0.125

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

/* IMPORTANT NOTE ABOUT POINT ARRAYS
 * they are stored with a stride of 4. 
 *  The first element is the channel (as a float)
 *  The other three are the position (x, y, z)
 */

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> npcarray;


inline void simd_err_func_helper(float* erf_i, v8sf delta, int bound, float ic, float cell_i, float dim_size){
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


void multithreaded_gaussian_erf_avx(size_t n_atoms, const float* points, GridSpec* o, float* tensor){
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
 

void gaussian_erf_avx_fit_points(size_t n_atoms, float* points, OutputSpec* o, float* tensor){
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

void datagen_consumer_avx_py(PyDataQueue* data_ptr){
    /* a consumer for multithreaded calculation */
    while(true){
        PyDataQueue& data = *data_ptr;
        PyConsumeInfo info;
        {
            std::unique_lock<std::mutex> lock(data.mtx);
            while(data.consume_queue.empty()){
                if(data.done) return;
                data.cv.wait(lock);
            }
            info = data.consume_queue.front();
            data.consume_queue.pop_front();
        }
        OutputSpec* out = &info.outs[0];
        size_t stride = out->H*out->W*out->D*out->N;
        for(int i = 0; i < info.n_molecules; i++){
            OutputSpec* out = &info.outs[i];
            gaussian_erf_avx_fit_points(info.molecule_sizes[i], info.molecules[i], out, info.tensor_out + stride*i);
        }
        {
            std::unique_lock<std::mutex> lock(data.mtx);
            data.amt_done++;
            data.cv.notify_all();
        }
    }
}

py::array_t<float> coord_to_grid(npcarray points,
                   const float width, const float height, const float depth,
                   const int grid_size, const int num_channels, float variance = 0.04){
    /* add a single point to a grid tensor */
    

    
    size_t n_atoms = points.shape(0);
    const float* point_ptr = (float*)points.request().ptr;


    std::vector<ssize_t> grid_shape = {(ssize_t)num_channels, grid_size, grid_size, grid_size};
    npcarray tensor = py::array_t<float>(grid_shape);
    float* out_tensor = (float*)tensor.request().ptr;

    memset(out_tensor, 0, num_channels*grid_size*grid_size*grid_size*sizeof(float));
 
    GridSpec out = {
        .width = width,
        .height = height,
        .depth = depth,
        .variance = variance,
        .grid_size = grid_size,
        .num_molecules = n_atoms
    };
    multithreaded_gaussian_erf_avx(n_atoms, point_ptr, &out, out_tensor);
    return tensor;
}

void add_to_grid_c(py::list molecules, npcarray tensor, npcarray* extents,
            float variance = 0.04){
    /* c interface for the add_to_grid function.
     * adds provided molecules to the provided tensors
     * with optional user-specified extents
     */

    tensor.unchecked<5>();

    int W = tensor.shape(4);
    int H = tensor.shape(3);
    int D = tensor.shape(2);
    int N = tensor.shape(1);
    ssize_t M = tensor.shape(0);

    // printf("w: %f, h: %f, d: %f, W: %d, H: %d, D: %d\n", width, height, depth, W, H, D);

    std::vector<int> array_sizes;
    std::vector<float*> array_pointers;
    OutputSpec out_specs[M];

    for(py::handle m : molecules){
        npcarray points = py::cast<npcarray>(m);
        ssize_t s = points.shape(0);
        ssize_t s2 = points.shape(1);
        array_sizes.push_back(s);

        float* point_ptr = (float*)points.request().ptr;
        float* f = new float[s*s2];
        memcpy(f, point_ptr, s*s2*sizeof(float));
        array_pointers.push_back(f);
    }

    long chunk_size = 8;
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    PyDataQueue data_queue;
    data_queue.n_molecules = chunk_size;

    for(int i = 0; i < n_threads; i++){
        threads.push_back(std::thread(datagen_consumer_avx_py, &data_queue));
    }

    size_t stride = H*W*D*N;
    float* out_tensor = (float*)tensor.request().ptr;
    float* extent_ptr = nullptr;
    if(extents){
        extent_ptr = (float*)extents->request().ptr;
    }

    // printf("-=-=-=-=-=-=- Before -=-=-=-=-=-=-\n");
    // for(int i = 0; i < N; i += 1){
        // display_tensor(W, H, D, out_tensor + stride*i, 4);
    // }

    for(int i = 0; i < (int)array_pointers.size(); i += chunk_size){
        PyConsumeInfo info;
        info.n_molecules = std::min(chunk_size, M - i);
        info.molecule_sizes = &array_sizes[i];
        info.molecules = &array_pointers[i];
        info.tensor_out = out_tensor + stride*i;
        for(int i2 = 0; i2 < chunk_size && i2 < info.n_molecules; i2++){
            out_specs[i+i2] = {
                .ext = {},
                .erf_inner_c = (float)(1/(SQRT_2*variance)),
                .erf_outer_c = 0.125,
                .W = W,
                .H = H,
                .D = D,
                .N = N
            };
            if(extent_ptr){
                float* a = &extent_ptr[(i+i2)*6];
                memcpy(out_specs[i+i2].ext, a, 6*sizeof(float));
            }
            else{
                get_grid_extent(info.molecule_sizes[i2], info.molecules[i2], out_specs[i+i2].ext);
            }
        }
        info.outs = &out_specs[i];
        {
            std::unique_lock<std::mutex> lock(data_queue.mtx);
            data_queue.consume_queue.push_back(info);
            data_queue.cv.notify_one();
        }
    }
    {
        std::unique_lock<std::mutex> lock(data_queue.mtx);
        data_queue.done = true;
        data_queue.cv.notify_all();
    }

    for(auto& tr : threads) tr.join();
    for(float* f : array_pointers) delete[] f;

    // printf("-=-=-=-=-=-=- After -=-=-=-=-=-=-\n");
    // for(int i = 0; i < N; i += 1){
        // display_tensor(W, H, D, out_tensor + stride*i, 4);
    // }
}

void add_to_grid(py::list molecules, npcarray tensor,
            float variance = 0.04){
    /* python wrapper */
    add_to_grid_c(molecules, tensor, nullptr, variance);
}

py::array_t<float> generate_grid_multithreaded_c(py::list molecules, npcarray* extents,
            int W = 32, int H = 32, int D = 32, int N = 1,
            float variance = 0.04){
    /*
     * creates a new tensor and runs add_to grid. Returns the tensor.
     */
    size_t M = molecules.size();
    std::vector<ssize_t> grid_shape = {(ssize_t)M, N, D, H, W};
    std::vector<ssize_t> grid_strides = {};
    npcarray grid = py::array_t<float>(grid_shape);

    float* out_tensor = (float*)grid.request().ptr;
    memset(out_tensor, 0, M*N*W*H*D*sizeof(float));
    add_to_grid_c(molecules, grid, extents, variance);

    return grid;
}


py::array_t<float> generate_grid_multithreaded_extents(py::list molecules, npcarray extents,
            int W = 32, int H = 32, int D = 32, int N = 1,
            float variance = 0.04){
    /* python wrapper */
    return generate_grid_multithreaded_c(molecules, &extents, W, H, D, N, variance);
}

py::array_t<float> generate_grid_multithreaded(py::list molecules,
            int W = 32, int H = 32, int D = 32, int N = 1,
            float variance = 0.04){
    /* python wrapper */
    return generate_grid_multithreaded_c(molecules, nullptr, W, H, D, N, variance);
}

void display_tensor_py(npcarray tensor, int show_max = 10){
    /* displays a tensor in ascii for debugging */
    int W = tensor.shape(4);
    int H = tensor.shape(3);
    int D = tensor.shape(2);
    int stride = W*H*D;
    float* ptr = (float*)tensor.request().ptr;
    ssize_t size = tensor.size();
    printf("Total size: %ld\n", size);
    for(ssize_t i = 0, j = 0; i < size && j < show_max; i += stride, j++){
        printf("%lu \tout of \t%ld (at %p)\n", i, size, ptr + i);
        display_tensor_xy(W, H, D, ptr + i, 4);
    }
}

void validate_tensor_py(npcarray tensor, bool check_zero = false){
    /* does some checks on the tensor to check it looks descent */
    int W = tensor.shape(4);
    int H = tensor.shape(3);
    int D = tensor.shape(2);
    int N = tensor.shape(1);
    int M = tensor.shape(0);
    float* ptr = (float*)tensor.request().ptr;
    validate_tensor(M, N, W, H, D, ptr, check_zero);
}


PYBIND11_MODULE(GridGenerator, m) {
    m.doc() = "Generate grids from point clouds";
    m.def("coord_to_grid", &coord_to_grid,
          "Convert a single 4-D point-cloud to grid",
          py::arg("points"),
          py::arg("width"), py::arg("height"), py::arg("depth"),
          py::arg("grid_size"), py::arg("num_channels"),py::arg("variance") = 0.04);
    m.def("generate_grid", &generate_grid_multithreaded, 
            "Generate a grid from a list of Nx4 numpy arrays", 
            py::arg("molecules"), py::arg("W") = 32, py::arg("H") = 32, py::arg("D") = 32, py::arg("N") = 1,
            py::arg("variance") = 0.04);
    m.def("generate_grid_extents", &generate_grid_multithreaded_extents, 
            "Generate a grid from a list of Nx4 numpy arrays",  
            py::arg("molecules"), py::arg("extents"), py::arg("W") = 32, py::arg("H") = 32, py::arg("D") = 32, py::arg("N") = 1,
            py::arg("variance") = 0.04);
    m.def("add_to_grid", &add_to_grid, 
            "Add molecule data to a pre-existing grid", 
            py::arg("molecules"), py::arg("tensor"),
            py::arg("variance") = 0.04);
    m.def("display_tensor", &display_tensor_py, 
            "Display the tensor in an ascii graph depiction", 
            py::arg("tensor"), py::arg("count") = 10);
    m.def("validate_tensor", &validate_tensor_py, 
            "Validate that the data in a tensor looks \"right\"", 
            py::arg("tensor"), py::arg("check_zero") = false);
    #ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = "dev";
    #endif
}

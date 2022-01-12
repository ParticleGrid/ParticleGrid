#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>

#include <omp.h>

#include "cgridgen.h"
#include "generate.h"
#include "tensor_handling.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

/* IMPORTANT NOTE ABOUT POINT ARRAYS
 * they are stored with a stride of 4. 
 *  The first element is the channel (as a float)
 *  The other three are the position (x, y, z)
 */

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> npcarray;

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
            gaussian_erf(info.molecule_sizes[i], info.molecules[i], out, info.tensor_out + stride*i);
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
    OutputSpec out; 
    float ext[2][3] = { 
        {0, 0, 0},
        {width, height, depth}
    }
    fill_output_spec(&out, grid_size, grid_size, grid_size, num_channels);
    gaussian_erf(n_atoms, point_ptr, &out, out_tensor);
    return tensor;
}

void add_to_grid_c(py::list molecules, npcarray tensor, npcarray* extents,
            float variance = 0.04, int n_threads = N_THREADS){
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
    if(n_threads == 0) n_threads = std::thread::hardware_concurrency(); 
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

    for(int i = 0; i < (int)array_pointers.size(); i += chunk_size){
        PyConsumeInfo info;
        info.n_molecules = std::min(chunk_size, M - i);
        info.molecule_sizes = &array_sizes[i];
        info.molecules = &array_pointers[i];
        info.tensor_out = out_tensor + stride*i;
        for(int i2 = 0; i2 < chunk_size && i2 < info.n_molecules; i2++){
            float* given_extent = nullptr;
            if(extent_ptr) {
                given_extent = &extent_ptr[(i+i2)*6];
            }
            else{
                get_grid_extent(info.molecule_sizes[i2], info.molecules[i2], ospec->ext);
            }
            fill_output_spec(&out_specs[i+i2], 
                    variance,
                    W, H, D, N, 
                    given_extent);
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

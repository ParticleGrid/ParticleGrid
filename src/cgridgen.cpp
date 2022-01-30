#include <cstdio>
#include <vector>
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
    };
    fill_output_spec(&out, variance, grid_size, grid_size, grid_size, num_channels, (float*)ext);
    gaussian_erf(n_atoms, point_ptr, &out, out_tensor);
    return tensor;
}

void add_to_grid_c(py::list molecules, const std::vector<ssize_t>& shape, const std::vector<ssize_t>& strides, float* tensor, float* extents,
            float variance = 0.04){
    /* c interface for the add_to_grid function.
     * adds provided molecules to the provided tensors
     * with optional user-specified extents
     */

    int W = shape[4];
    int H = shape[3];
    int D = shape[2];
    int N = shape[1];
    // ssize_t M = tensor.shape(0);

    size_t stride = strides[0]/sizeof(float);
    float* out_tensor = tensor;
    float* extent_ptr = extents;
    size_t i = 0;
    for(py::handle m : molecules){
        npcarray points = py::cast<npcarray>(m);
        ssize_t shape = points.shape(0);

        int num_atoms = shape;
        float* atom_ptr = (float*)points.request().ptr;
        float* tensor_out = out_tensor + stride*i;

        OutputSpec out_spec;
        float* given_extent = nullptr;
        float ext[2][3];
        if(extent_ptr) {
            given_extent = &extent_ptr[i*6];
        }
        else{
            get_grid_extent(num_atoms, atom_ptr, ext);
            given_extent = (float*)ext;
        }
        fill_output_spec(&out_spec, 
                variance,
                W, H, D, N, 
                given_extent);
        // printf("mol: %ld %ld\n", i, stride*i);
        gaussian_erf(num_atoms, atom_ptr, &out_spec, tensor_out);
        i++;
    }
}

void* optional_npcarray(npcarray* arr){
    if(!arr) return nullptr;
    return arr->request().ptr;
}

void add_to_grid_helper(py::list molecules, npcarray tensor, npcarray* extents,
            float variance = 0.04){
    /* python wrapper */
    tensor.unchecked<5>();
    ssize_t ndim = tensor.ndim();
    std::vector<ssize_t> shape(ndim);
    std::vector<ssize_t> stride(ndim);
    for(int i = 0; i < ndim; i++){
        shape[i] = tensor.shape(i);
        stride[i] = tensor.strides(i);
    }
    float* extents_float = (float*)optional_npcarray(extents);
    add_to_grid_c(molecules, shape, stride, (float*)tensor.request().ptr, extents_float, variance);
}

py::array_t<float> generate_grid_c(py::list molecules, npcarray* extents,
            int W = 32, int H = 32, int D = 32, int N = 1,
            float variance = 0.04){
    /*
     * creates a new tensor and runs add_to_grid. Returns the tensor.
     */
    size_t M = molecules.size();
    std::vector<ssize_t> grid_shape = {(ssize_t)M, N, D, H, W};
    size_t x_stride = W;
    #ifdef V8F
    size_t miss = W%8;
    if(miss) x_stride += (8-miss);
    #endif
    std::vector<ssize_t> grid_strides = {N, D, H, (ssize_t)x_stride, (ssize_t)sizeof(float)};
    for(int i = grid_strides.size() - 2; i >= 0; i--) {
        grid_strides[i] *= grid_strides[i+1];
        // printf("%ld\n", grid_strides[i]);
    }
    float* tmp = nullptr;
    if(x_stride != (size_t)W){
        tmp = (float*)calloc(M, grid_strides[0]);
        float* extents_float = (float*)optional_npcarray(extents);
        add_to_grid_c(molecules, grid_shape, grid_strides, tmp, extents_float, variance);
    }
    npcarray grid = py::array_t<float>(grid_shape);
    /*
    npcarray grid = py::array_t<float>(py::buffer_info(
        tmp, // data (copied from here if not null)
        sizeof(float), // item size
        py::format_descriptor<float>::format(),
        grid_shape.size(), // number of dimensions
        grid_shape,
        grid_strides
    ));
    */

    if(tmp) {
        free(tmp);
    }
    else{
        float* out_tensor = (float*)grid.request().ptr;
        memset(out_tensor, 0, M*grid.strides(0));
        add_to_grid_helper(molecules, grid, extents, variance);
    }

    return grid;
}

py::array_t<float> generate_grid_extents(py::list molecules, npcarray extents,
            int W = 32, int H = 32, int D = 32, int N = 1,
            float variance = 0.04){
    /* python wrapper, with extents provided */
    return generate_grid_c(molecules, &extents, W, H, D, N, variance);
}

py::array_t<float> generate_grid(py::list molecules,
            int W = 32, int H = 32, int D = 32, int N = 1,
            float variance = 0.04){
    /* python wrapper */
    return generate_grid_c(molecules, nullptr, W, H, D, N, variance);
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

PYBIND11_MODULE(ParticleGrid, m) {
    m.doc() = "Generate grids from point clouds";
    m.def("coord_to_grid", &coord_to_grid,
          "Convert a single 4-D point-cloud to grid",
          py::arg("points"),
          py::arg("width"), py::arg("height"), py::arg("depth"),
          py::arg("grid_size"), py::arg("num_channels"),py::arg("variance") = 0.04);
    m.def("generate_grid", &generate_grid, 
            "Generate a grid from a list of Nx4 numpy arrays", 
            py::arg("molecules"), py::arg("W") = 32, py::arg("H") = 32, py::arg("D") = 32, py::arg("N") = 1,
            py::arg("variance") = 0.04);
    m.def("generate_grid_extents", &generate_grid_extents, 
            "Generate a grid from a list of Nx4 numpy arrays, with provided extents",  
            py::arg("molecules"), py::arg("extents"), py::arg("W") = 32, py::arg("H") = 32, py::arg("D") = 32, py::arg("N") = 1,
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


#include <cstdio>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "tensor_handling.hpp"
#include "cgridgen.h"
#include "generate.h"

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> npcarray;

template<class T>
T round_up(T val, ssize_t to){
    ssize_t miss = val % to;
    if(miss) val += (to - miss);
    return val;
}

std::vector<ssize_t> strides_for(const std::vector<ssize_t>& shape, size_t round_to = 8) {
    #ifdef V8F
    ssize_t x_stride = round_up(shape.back(), round_to);
    #else
    ssize_t x_stride = shape.back();
    #endif
    std::vector<ssize_t> grid_strides = {};
    for(size_t i = 1; i < shape.size(); i++){
        grid_strides.push_back(shape[i]);
    }
    grid_strides[grid_strides.size()-1] = x_stride;
    grid_strides.push_back(sizeof(float));
    for(ssize_t i = grid_strides.size() - 2; i >= 0; i--) {
        grid_strides[i] *= grid_strides[i+1];
    }
    return grid_strides;
}

void add_to_grid_c(const ssize_t num_points, const float* points, 
            const std::vector<ssize_t>& shape, const std::vector<ssize_t>& strides, 
            float* tensor, float* extent,
            float variance,
            Options* ptr_options){
    /* c interface for the add_to_grid function.
     * adds provided points to the provided tensor
     * with optional user-specified extents
     */

    Options options;
    if(ptr_options){
        options = *ptr_options;
    }
    int ndim = shape.size();
    int W = shape[ndim-1];
    int H = shape[ndim-2];
    int D = shape[ndim-3];
    int N = shape[ndim-4];

    OutputSpec out_spec;
    float* given_extent = nullptr;
    float ext[2][3];
    if(extent) {
        given_extent = extent;
    }
    else{
        get_grid_extent(num_points, points, ext);
        given_extent = (float*)ext;
    }
    fill_output_spec(&out_spec, 
            variance,
            // TODO make the following just a pointer/vec of four ints/ssize_t
            W, H, D, N, 
            given_extent,
            options);
    gaussian_erf(num_points, points, &out_spec, tensor);
}

py::array_t<float> create_point_grid_c(const npcarray& points,
                   const ssize_t W, const ssize_t H, const ssize_t D, const int num_channels, 
                   float* extent,
                   float variance,
                   Options* options){
    /* return a grid representing the 4D array `points` (channel, W, H, D)*/
    size_t num_points = points.shape(0);
    const float* ptr_points = (float*)points.request().ptr;
    std::vector<ssize_t> grid_shape = {(ssize_t)num_channels, D, H, W};
    std::vector<ssize_t> grid_strides = strides_for(grid_shape);
    float* tmp = nullptr;
    if(grid_strides[0] != strides_for(grid_shape, 1)[0]){
        printf("DOING A COPY\n");
        tmp = (float*)calloc(num_channels, grid_strides[0]);
        add_to_grid_c(num_points, ptr_points, grid_shape, grid_strides, tmp, extent, variance, options);
    }
    // npcarray grid = py::array_t<float>(grid_shape);
    npcarray grid = py::array_t<float>(py::buffer_info(
        tmp, // data (copied from here if not null)
        sizeof(float), // item size
        py::format_descriptor<float>::format(),
        grid_shape.size(), // number of dimensions
        grid_shape,
        grid_strides
    ));
    if(!tmp){
        float* ptr_tensor = (float*)grid.request().ptr;
        memset(ptr_tensor, 0, num_channels * grid_strides[0]);
        add_to_grid_c(num_points, ptr_points, grid_shape, grid_strides, ptr_tensor, extent, variance, options);
    }
    return grid;
}

void onto_grid(npcarray& tensor, npcarray& points, const int grid_size, const int num_channels,
            float variance) {
    std::vector<ssize_t> grid_shape = {(ssize_t)num_channels, grid_size, grid_size, grid_size};
    std::vector<ssize_t> grid_strides = strides_for(grid_shape);
    add_to_grid_c(points.shape(0), (float*)points.request().ptr, grid_shape, grid_strides, (float*)tensor.request().ptr, nullptr, variance, nullptr);
}

py::array_t<float> molecule_grid(npcarray points, const int grid_size, const int num_channels,
            float variance) {
    /*float ext[2][3] = {
        {0, 0, 0},
        {width, height, depth},
    };*/
    return create_point_grid_c(points, grid_size, grid_size, grid_size, num_channels, nullptr, variance, nullptr);
}
void display_tensor_py(npcarray tensor, int show_max = 10){
    /* displays a tensor in ascii for debugging */
    int W = tensor.shape(3);
    int H = tensor.shape(2);
    int D = tensor.shape(1);
    int stride = W*H*D;
    float* ptr = (float*)tensor.request().ptr;
    ssize_t size = tensor.size();
    printf("Tensor %p Total size: %ld\n", ptr, size);
    display_tensor_xy(W, H, D, ptr, 4);
}
size_t numpy_pointer(npcarray& tensor) {
    return (size_t)tensor.request().ptr;
}

PYBIND11_MODULE(GridGenerator, m) {
    m.doc() = "Generate grids from point clouds";
    m.def("onto_grid", &onto_grid,
          "Convert a single 4-D point-cloud to grid",
          py::arg("tensor"), py::arg("points"), py::arg("grid_size"), py::arg("num_channels"),
          py::arg("variance") = 0.04);
    m.def("molecule_grid", &molecule_grid,
          "Convert a single 4-D point-cloud to grid",
          py::arg("points"), py::arg("grid_size"), py::arg("num_channels"),
          py::arg("variance") = 0.04);
    m.def("display_tensor", &display_tensor_py, 
            "Display the tensor in an ascii graph depiction", 
            py::arg("tensor"), py::arg("count") = 10);
    #ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = "dev";
    #endif
}

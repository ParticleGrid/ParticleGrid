
#include <cstdio>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "tensor_handling.hpp"
#include "cgridgen.h"
#include "generate.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> npcarray;

void add_to_grid_c(
    float *tensor,
    const ssize_t num_points, const float *points,
    ssize_t num_dim,
    const ssize_t *shape, const ssize_t *strides,
    float *extent,
    float variance,
    Options *options)
{
    /* c interface for the add_to_grid function.
     * adds provided points to the provided tensor
     * with optional user-specified extents
     */

    OutputSpec out_spec;
    fill_output_spec(&out_spec,
                     num_points, points,
                     num_dim, shape, strides,
                     variance,
                     extent,
                     options);
    gaussian_erf(num_points, points, &out_spec, tensor);
}

py::array_t<float> create_point_grid_c(const npcarray &points,
                                       const ssize_t W, const ssize_t H, const ssize_t D, const int num_channels,
                                       float *extent,
                                       float variance,
                                       Options *options)
{
    /* return a grid representing the 4D array `points` (channel, W, H, D)*/
    size_t num_points = points.shape(0);
    const float *ptr_points = (float *)points.request().ptr;
    std::vector<ssize_t> grid_shape = {(ssize_t)num_channels, D, H, W};
    npcarray grid = py::array_t<float>(grid_shape);
    float *ptr_tensor = (float *)grid.request().ptr;
    memset(ptr_tensor, 0, grid.size() * sizeof(float));
    add_to_grid_c(ptr_tensor, num_points, ptr_points, grid.ndim(), grid.shape(), grid.strides(), extent, variance, options);
    return grid;
}

py::array_t<float> molecule_grid(npcarray points, const int grid_size, const int num_channels,
                                 float variance)
{
    /* convert a single molecule to a grid */
    return create_point_grid_c(points, grid_size, grid_size, grid_size, num_channels, nullptr, variance, nullptr);
}

py::array_t<float> list_grid_shape(py::list molecules, const std::vector<ssize_t> &shape, const int num_channels, float variance)
{
    /* convert a list of molecules to a grid */
    size_t M = molecules.size();
    std::vector<ssize_t> grid_shape = {(ssize_t)M, (ssize_t)num_channels, shape[0], shape[1], shape[2]};
    npcarray grid = py::array_t<float>(grid_shape);
    size_t i = 0;
    float *tensor = (float *)grid.request().ptr;
    ssize_t stride = grid.strides(0) / sizeof(float);
    for (py::handle m : molecules)
    {
        npcarray points = py::cast<npcarray>(m);
        add_to_grid_c(tensor + i * stride,
                      points.size() / 4,
                      (float *)points.request().ptr,
                      grid_shape.size() - 1,
                      grid.shape() + 1,
                      grid.strides() + 1,
                      nullptr,
                      variance,
                      nullptr);
        i++;
    }
    return grid;
}

py::array_t<float> list_grid(py::list molecules, ssize_t grid_shape, const int num_channels, float variance)
{
    /* convert a list of molecules to a grid */
    return list_grid_shape(molecules, {grid_shape, grid_shape, grid_shape}, num_channels, variance);
}

py::array_t<float> generate_grid(py::list molecules, const int W, const int H, const int D, const int num_channels, float variance)
{
    /* convert a list of molecules to a grid */
    return list_grid_shape(molecules, {D, H, W}, num_channels, variance);
}

py::array_t<float> coord_to_grid(npcarray points,
                                 const float width, const float height, const float depth,
                                 const int grid_size, const int num_channels, float variance = 0.04)
{
    /* add a single molecule to a grid tensor, with extents specified */
    float ext[2][3] = {
        {0, 0, 0},
        {width, height, depth}};
    return create_point_grid_c(points, grid_size, grid_size, grid_size, num_channels, (float *)ext, variance, nullptr);
}

void display_tensor_py(npcarray tensor, int show_max = 10)
{
    /* displays a tensor in ascii for debugging */
    int W = tensor.shape(3);
    int H = tensor.shape(2);
    int D = tensor.shape(1);
    float *ptr = (float *)tensor.request().ptr;
    ssize_t size = tensor.size();
    printf("Tensor %p Total size: %ld\n", ptr, size);
    display_tensor_xy(W, H, D, ptr, 4);
}

// To do: add sub-modules as described here: https://github.com/pybind/pybind11/discussions/4027

PYBIND11_MODULE(GridGenerator, m)
{
    m.doc() = "Generate grids from point clouds";
    m.def("molecule_grid", &molecule_grid,
          "Convert a single 4-D point-cloud to grid",
          py::arg("points"), py::arg("grid_size"), py::arg("num_channels"),
          py::arg("variance") = 0.04);
    m.def("list_grid", &list_grid,
          "Convert a single 4-D point-cloud to grid",
          py::arg("molecules"), py::arg("grid_size"), py::arg("num_channels"),
          py::arg("variance") = 0.04);
    m.def("generate_grid", &generate_grid,
          "Generate a grid from a list of Nx4 numpy arrays",
          py::arg("molecules"), py::arg("W") = 32, py::arg("H") = 32, py::arg("D") = 32, py::arg("N") = 1,
          py::arg("variance") = 0.04);
    m.def("coord_to_grid", &coord_to_grid,
          "Convert a single 4-D point-cloud to grid",
          py::arg("points"),
          py::arg("width"), py::arg("height"), py::arg("depth"),
          py::arg("grid_size"), py::arg("num_channels"), py::arg("variance") = 0.04);
    m.def("display_tensor", &display_tensor_py,
          "Display the tensor in an ascii graph depiction",
          py::arg("tensor"), py::arg("count") = 10);
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

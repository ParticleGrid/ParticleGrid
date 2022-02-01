#include "ParticleGrid.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <Python.h>

namespace py = pybind11;
template <typename T>
using NPCarray = py::array_t<T, py::array::c_style | py::array::forcecast>;

#define SINGLE_MOLECULE NPCarray<float>, float, float, float, int, int, float
#define BATCHED_MOLECULE py::list, NPCArray<float>, NPCArray<float>, NPCArray<float>, NPCArray<int>, NPCArray<int>, NPCArray<float>


// To do: We could use variadic templates here and move parameters up on to the Gridder's themselves. 



NPCarray ERFGridGenerator(py::list molecules, 
                          NPCArray<float> widths, 
                          NPCArray<float> heights, 
                          NPCArray<float> depths, 
                          int grid_size, 
                          int num_channels, 
                          NPCArray<float> variances){
  ERFGrid<float, true, CPU> gridder();

  gridder.set_grid_params();
  
  return gridder.get_grid();
}

NPCarray ERFGridGenerator(NPCarray<float> atoms, 
                          float widths, 
                          float heights, 
                          float depths, 
                          int grid_size, 
                          int num_channels, 
                          float variances){
  ERFGrid<float, false, CPU> gridder();

  gridder.set_grid_params();
  
  return gridder.get_grid();
}




PYBIND11_MODULE(GridGenerator, m){
  m.doc() = "Generate grids from molecular point clouds";
  m.def("ERFGridGenerator", py::overload_cast<SINGLE_MOLECULE>(&ERFGridGenerator), "Generates ERF grid for a single molecule or batch of molecules");
  m.def("ERFGridGenerator", py::overload_cast<BATCHED_MOLECULE>(&ERFGridGenerator), "Generates ERF grid for a single molecule or batch of molecules");
}
  
  /* Commenting out but for future
  m.def("WeightedERFGridGenerator", 
        py::overload_cast<SINGLE_MOLECULE>(&WeightedERFGridGenerator), 
        "Generates weighted ERF grid for a single or batch molecule");
  m.def("WeightedERFGridGenerator",
        py::overload_cast<BATCHED_MOLECULE>(&WeightedERFGridGenerator), 
        "Generates weighted ERF grid for a single or batch molecule");

  m.def("PeriodicERFGridGenerator",
        py::overload_cast<SINGLE_MOLECULE>(&PeriodicERFGridGenerator),
        "Generates periodic ERF grid for a single or batch molecule");
  m.def("PeriodicERFGridGenerator",
        py::overload_cast<BATCHED_MOLECULE>(&PeriodicERFGridGenerator),
        "Generates periodic ERF grid for a single or batch molecule");
  
  m.def("BinaryERFGridGenerator",
        py::overload_cast<SINGLE_MOLECULE>(&PeriodicERFGridGenerator),
        "Generates periodic ERF grid for a single or batch molecule");
  m.def("BinaryERFGridGenerator",
        py::overload_cast<BATCHED_MOLECULE>(&BinaryERFGridGenerator),
        "Generates a binarized ERF grid for a single or batch molecule"); */
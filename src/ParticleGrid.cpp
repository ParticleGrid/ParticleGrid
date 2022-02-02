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


PYBIND11_MODULE(GridGenerator, m){
  m.doc() = "Generate grids from molecular point clouds";
  m.def("ERFGridGenerator",
        py::overload_cast<SINGLE_MOLECULE>(&ERFGridGenerator),
        "Generates ERF grid for a single molecule or batch of molecules");
  m.def("ERFGridGenerator",
        py::overload_cast<BATCHED_MOLECULE>(&ERFGridGenerator),
        "Generates ERF grid for a single molecule or batch of molecules");

  m.def("BinaryERFGridGenerator",
        py::overload_cast<SINGLE_MOLECULE>(&PeriodicERFGridGenerator),
        "Generates periodic ERF grid for a single or batch molecule");
  m.def("BinaryERFGridGenerator",
        py::overload_cast<BATCHED_MOLECULE>(&BinaryERFGridGenerator),
        "Generates a binarized ERF grid for a single or batch molecule"); 
}
  
  /* Commenting out but for future

  m.def("PeriodicERFGridGenerator",
        py::overload_cast<SINGLE_MOLECULE>(&PeriodicERFGridGenerator),
        "Generates periodic ERF grid for a single or batch molecule");
  m.def("PeriodicERFGridGenerator",
        py::overload_cast<BATCHED_MOLECULE>(&PeriodicERFGridGenerator),
        "Generates periodic ERF grid for a single or batch molecule");
  
 */
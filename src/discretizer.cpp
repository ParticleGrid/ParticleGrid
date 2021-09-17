#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include <cmath>

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> npcarray;



inline float
squared_error_forward(const float* original_grid, const float* candidate_grid, const int num_grid_points){
  float total = 0;
  for (auto i = 0; i < num_grid_points; ++i){
    const auto& err = (candidate_grid[i] - original_grid[i]);
    total += err * err; 
  }
  return total;
}

void 
squared_error_backward(const float* original_grid, 
                       const float* candidate_grid,
                       float*  gradient_wrt_candidate,
                       const int& num_grid_points,
                       const float& threshold,
                       bool& stop_backprop){
  float errs[num_grid_points];
  float total = 0;

  for (auto i = 0; i < num_grid_points; ++i){
    const auto& err = (candidate_grid[i] - original_grid[i]);
    total += err * err; 
    errs[i] = err; // Cache error
  }

  // if (total < threshold){
  //   // No need to calculate backprop as the candidate grid
  //   // and the original grid are close enough according to
  //   // the threshold set
  //   stop_backprop = true;
  //   return;
  // }
}

// void 
// coord_to_grid_forward(const float* points,
//                       float* grid,
//                       ){

// }


py::array_t<float> 
estimate_coords(const float* grid,
                const int num_channels, 
                const int offset){

  float total = 0;

  std::vector<float> coord_data;

  float channel_total;
  int n_atoms_in_channel;

  for (auto channel = 0; channel < num_channels; ++channel){
    
    channel_total = 0;

    for (auto grid_point_index = 0; grid_point_index < offset; ++grid_point_index){

      channel_total += grid[channel*offset + grid_point_index];
      total += grid[channel*offset + grid_point_index];
    }

    n_atoms_in_channel = static_cast<int>(std::ceil(channel_total));

  }

  int n_atoms = static_cast<int>(std::ceil(total));

  std::vector<ssize_t> coords_shape = {(ssize_t)n_atoms, 4};
  npcarray coords = py::array_t<float>();
  return coords;
}

py::array_t<float> 
grid_to_coords(npcarray grid){

  // Following line assumes a 4D grid, which is while valid
  // not particularly extensible

  const std::vector<ssize_t> grid_dims{grid.shape(), grid.shape()+4}; 
  const int num_channels = grid_dims[0];

  // The offset is the number of grid points per channel. It's used as a stride for 
  // traversing through the grid pointer

  const int offset = std::accumulate(grid_dims.begin()+1, grid_dims.end(), 1, std::multiplies<int>());

  const float* grid_ptr = (float*)grid.request().ptr;


  auto estimated_coords = estimate_coords(grid_ptr, num_channels, offset);

  bool needs_optimize = true;
  // while(0){
    
  // }
  return estimated_coords;
}



PYBIND11_MODULE(Discretizer, m) {
  m.doc() = "Retrieve atom set from grids";
  m.def("grid_to_coords", &grid_to_coords, 
        "Convert grid to a set of 4P point cloud", 
        py::arg("grid"));
}
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
squared_error_fused(const float* original_grid, 
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

  if (total < threshold){
    // No need to calculate backprop as the candidate grid
    // and the original grid are close enough according to
    // the threshold set
    stop_backprop = true;
    return;
  }



}
py::array_t<float> 
estimate_coords(float* grid,
                int num_channels, 
                int offset){

  float total = 0;

  std::vector<float> coord_data;

  float channel_total;
  int n_atoms_in_channel;

  for (auto channel = 0; i < num_channels; ++channel){
    
    channel_total = 0;

    for (auto grid_point_index = 0; grid_point_index < offset; ++grid_point_index){

      channel_total += grid[channel*offest + grid_point_index];
      total += grid[channel*offest + grid_point_index];
    }

    n_atoms_in_channel = std::static_cast<int>(std::ceil(channel_total));

  }

  int n_atoms = std::static_cast<int>(std::ceil(total));

  std::vector<ssize_t> coords_shape = {(ssize_t)n_atoms, 4};
  npcarray coords = py::array_t<float>();
}

py::array_t<float> 
grid_to_coords(npcarray grid){
  std::vector<ssize_t> grid_dims  = grid.shape(); 
  
  int num_channels = grid_dims[0];

  // The offset is the number of grid points per channel. It's used as a stride for 
  // traversing through the grid pointer

  int offset = std::accumulate(grid_dims.begin()+1, grid_dims.end(), 1, std::multiplies<int>());

  auto& initial_estimate = num_atoms(grid, num_channel, num_grid_points);

  bool needs_optimize = true;

  while(needs_optimize){

  }

}



PYBIND11_MODULE(Discretizer, m) {
  m.doc() = "Retrieve atom set from grids";
  m.def("grid_to_coords", &grid_to_coords, 
        "Convert grid to a set of 4P point cloud", 
        py::arg("grid"));
}
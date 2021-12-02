
#include "discretizer.hpp"

#define SQRT_2 1.41421356237
#define INTEGRAL_NORMALIZATION_3D 0.125

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> npcarray;

py::array_t<float> 
grid_to_coords(npcarray grid,
               const float variance,
               const float threshold, 
               const float max_iterations){

  // Following line assumes a 4D grid, which is while valid
  // not particularly extensible

  const std::vector<ssize_t> grid_dims{grid.shape(), grid.shape()+4}; 
  const int num_channels = grid_dims[0];
  const int grid_size = grid_dims[1];

  // The offset is the number of grid points per channel. It's used as a stride for 
  // traversing through the grid pointer

  const int offset = std::accumulate(grid_dims.begin()+1, grid_dims.end(), 1, std::multiplies<int>());
  const int num_grid_points = num_channels * offset;

  py::print("Num Channels",num_channels, "Offset ",offset);

  const float* grid_ptr = (float*)grid.request().ptr;


  npcarray estimated_coords; 
  estimate_coords(grid_ptr, num_channels, offset, estimated_coords);

  bool needs_optimize = true;
  auto iteration = 0; 

  // Set up containers for SGD

  npcarray predicted_grid = py::array_t<float>(grid_dims);  
  float* predicted_grid_ptr = (float*)predicted_grid.request().ptr;
  std::memset(predicted_grid_ptr, 0, num_grid_points);

  float grid_points_grad[num_grid_points];
  float estimated_coords_grad_ptr[estimated_coords.shape(0)*3];

  // Set up constants for grid

  float oc = INTEGRAL_NORMALIZATION_3D;
  float ic = (float)(1/(SQRT_2 * variance));
  const float lr = 0.01;
  while(iteration < max_iterations){
    // Generate the grid with the estimated points
    coord_to_grid_forward(estimated_coords,
                          grid_size,
                          num_channels,
                          ic,
                          oc,
                          predicted_grid_ptr);

    // Calculate the error between the predicted 
    // grid and the original grid.
    // Concurrently calculate the gradient with 
    // respect the grid elements and signal if backward 
    // pass is needed
    squared_error_backward(grid_ptr,
                           predicted_grid_ptr,
                           grid_points_grad,
                           num_grid_points,
                           threshold,
                           needs_optimize);
    if (!needs_optimize){
      py::print("Stopping updates");
      break;
    }
    // If the backward pass is needed, update the  
    // coordinates with the calculated gradients 

    generate_grid_backward(estimated_coords,
                           grid_points_grad,
                           estimated_coords_grad_ptr,
                           grid_size,
                           variance);

    update_coords(estimated_coords,
                  estimated_coords_grad_ptr,
                  lr);
    iteration++;
  }
  return estimated_coords;  
}

py::array_t<float> 
optimize_coords(npcarray grid,
               const npcarray estimated_coords,
               const float variance,
               const float threshold, 
               const float max_iterations){
  // Following line assumes a 4D grid, which is while valid
  // not particularly extensible

  const std::vector<ssize_t> grid_dims{grid.shape(), grid.shape()+4}; 
  const int num_channels = grid_dims[0];
  const int grid_size = grid_dims[1];

  const int num_atoms = estimated_coords.shape(0);

  std::vector<ssize_t> coords_shape = {(ssize_t)num_atoms, 4};
  npcarray coords = py::array_t<float>(coords_shape);
  auto* coords_ptr = (float*)coords.request().ptr;

  // Copy over the estimated data
  std::memcpy(coords_ptr, (float*)estimated_coords.request().ptr, num_atoms*4*sizeof(float));

  // The offset is the number of grid points per channel. It's used as a stride for 
  // traversing through the grid pointer

  const int offset = std::accumulate(grid_dims.begin()+1, grid_dims.end(), 1, std::multiplies<int>());
  const int num_grid_points = num_channels * offset;

  py::print("Num Channels",num_channels, "Offset ",offset);

  const float* grid_ptr = (float*)grid.request().ptr;

  bool needs_optimize = true;
  auto iteration = 0; 

  // Set up containers for SGD

  npcarray predicted_grid = py::array_t<float>(grid_dims);  
  float* predicted_grid_ptr = (float*)predicted_grid.request().ptr;
  std::memset(predicted_grid_ptr, 0, num_grid_points);

  float grid_points_grad[num_grid_points];
  float estimated_coords_grad_ptr[coords.shape(0)*3];

  // Set up constants for grid

  const float oc = INTEGRAL_NORMALIZATION_3D;
  const float ic = (float)(1/(SQRT_2 * variance));
  const float lr = 0.01;
  while(iteration < max_iterations){
    // Generate the grid with the estimated points
    coord_to_grid_forward(coords,
                          grid_size,
                          num_channels,
                          ic,
                          oc,
                          predicted_grid_ptr);

    // Calculate the error between the predicted 
    // grid and the original grid.
    // Concurrently calculate the gradient with 
    // respect the grid elements and signal if backward 
    // pass is needed
    squared_error_backward(grid_ptr,
                           predicted_grid_ptr,
                           grid_points_grad,
                           num_grid_points,
                           threshold,
                           needs_optimize);
    if (!needs_optimize){
      py::print("Stopping updates");
      break;
    }
    // If the backward pass is needed, update the  
    // coordinates with the calculated gradients 

    generate_grid_backward(coords,
                           grid_points_grad,
                           estimated_coords_grad_ptr,
                           grid_size,
                           variance);

    update_coords(coords,
                  estimated_coords_grad_ptr,
                  lr);
    iteration++;
  }
  return coords;
}

PYBIND11_MODULE(Discretizer, m) {
  m.doc() = "Retrieve atom set from grids";
  m.def("grid_to_coords", &grid_to_coords, 
        "Convert grid to a set of 4P point cloud", 
        py::arg("grid"),
        py::arg("variance") = 0.04,
        py::arg("tolerance") = 2.0e-4,
        py::arg("max_iterations") = 10000);
  m.def("optimize_coords", &optimize_coords,
        "Optimize a set of estimated coordinates to a grid",
        py::arg("grid"),
        py::arg("estimated_coords"),
        py::arg("variance"),
        py::arg("tolerance") = 2.0e-4,
        py::arg("max_iterations") = 10000);
}


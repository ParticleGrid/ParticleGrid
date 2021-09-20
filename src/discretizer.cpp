#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include <cmath>
#include <iostream>
#include <random>
#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp))
#define SQRT_2 1.41421356237
#define INTEGRAL_NORMALIZATION_3D 0.125

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
  float total = 0;

  for (auto i = 0; i < num_grid_points; ++i){
    const auto& err = (candidate_grid[i] - original_grid[i]);
    total += err * err; 
    gradient_wrt_candidate[i] = 2*err; // Cache error
  }

  if (total < threshold){
    // No need to calculate backprop as the candidate grid
    // and the original grid are close enough according to
    // the threshold set
    stop_backprop = true;
  }
}

void 
coord_to_grid_forward(const npcarray estimated_coords,
                      const int grid_size,
                      const int num_channels,
                      const float ic,
                      const float oc,
                      float* grid){
  const float cell_w = 1/ grid_size;
  const float cell_h = 1/ grid_size;
  const float cell_d = 1/ grid_size;

  // Generates a grid 

  const auto n_atoms = estimated_coords.shape(0);
  const auto stride = grid_size * grid_size * grid_size;

  float* points = (float*)estimated_coords.request().ptr;

  std::memset(grid, 0, stride*num_channels*sizeof(float));

  for (auto atom = 0; atom<n_atoms; ++atom){
    
    float erfx[grid_size+1];
    float erfy[grid_size+1];
    float erfz[grid_size+1];

    auto points_offset = atom*4; 
    const auto channel = (int)points[points_offset];
    const auto x = points[points_offset+1];
    const auto y = points[points_offset+2];
    const auto z = points[points_offset+3];

    for (auto i = 0; i < grid_size+1; ++i){
      auto left_bound = cell_w * i;
      auto delta_x_lower = (x - left_bound);
      auto delta_y_lower = (y - left_bound);
      auto delta_z_lower = (z - left_bound);

      erfx[i] = std::erf(delta_x_lower * ic);
      erfy[i] = std::erf(delta_y_lower * ic);
      erfz[i] = std::erf(delta_z_lower * ic);
    }

    for (auto i = 0; i < grid_size; ++i){
      erfx[i] = erfx[i+1] - erfx[i];
      erfy[i] = erfy[i+1] - erfy[i];
      erfz[i] = erfz[i+1] - erfz[i];
    }

    auto channel_offset = channel * (stride);

    for(auto k = 0; k < grid_size; ++k){
      auto z_erf = erfz[k] * oc;
      auto z_offset = channel_offset + k * (grid_size*grid_size);
      for(auto j = 0; j < grid_size; ++j){
        auto yz_erf = erfy[j] * z_erf;
        auto zy_offset = z_offset + j * (grid_size);
        for(auto i = 0; i < grid_size; ++i){
          auto id =  zy_offset + i;
          auto val = erfx[i] * yz_erf;
          grid[id] += val; 
        }
      }
    }
  }
}


void 
estimate_coords(const float* grid,
                const int num_channels, 
                const int offset,
                npcarray& coords){

  std::vector<float> coord_data;
  float total = 0;
  float channel_total;
  int n_atoms_in_channel;
  std::vector<int> atoms_per_channel(num_channels, 0);

  for (auto channel = 0; channel < num_channels; ++channel){
    channel_total = 0;
    for (auto grid_point_index = 0; grid_point_index < offset; ++grid_point_index){

      channel_total += grid[channel*offset + grid_point_index];
      total += grid[channel*offset + grid_point_index];
    }
    // The question remains whether round is a good way of doing this
    // Particularly with small sized grid, the loss in the erf values may  
    // be large enough such that the rounding produces incorrect 
    // number of atoms. Also possibly an issue with unpadded grids
    // Assume padded grids for now
    n_atoms_in_channel = static_cast<int>(std::round(channel_total));

    if (n_atoms_in_channel > 0){
      atoms_per_channel[channel] = n_atoms_in_channel;
    }
  }

  int n_atoms = static_cast<int>(std::round(total));
  // Initialize the numpy container with for the coordinates
  std::vector<ssize_t> coords_shape = {(ssize_t)n_atoms, 4};
  coords = py::array_t<float>(coords_shape);
  auto* coords_ptr = (float*)coords.request().ptr;

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 1.0); // Generates uniform random number between 0 and 1

  std::vector<float> flatten_estimated_coords (n_atoms*4, 0);

  auto atom_num = 0;
  for(auto i = 0; i < num_channels; ++i){
    // For each atom type
    if (atoms_per_channel[i] > 0){
      py::print("Channel ", i, " has ", atoms_per_channel[i], " atoms");
      // If the atom type exists
      for (auto k = 0; k < atoms_per_channel[i]; ++k){
        // For each of the atoms of this atom type
        auto _coord_offset_temp = atom_num * 4; 
        flatten_estimated_coords[_coord_offset_temp] = i; 
        for(auto j = 1; j < 4; ++j){
            flatten_estimated_coords[_coord_offset_temp+j] = dis(gen);
        } 
        ++atom_num;
      }
    }
  }
  py::print(atom_num, n_atoms);
  assertm(atom_num == n_atoms, "Sanity check failed for number of atoms");
  std::memcpy(coords_ptr, flatten_estimated_coords.data(), n_atoms*4*sizeof(float));
}

py::array_t<float> 
grid_to_coords(npcarray grid,
               const float variance){

  // Following line assumes a 4D grid, which is while valid
  // not particularly extensible

  const std::vector<ssize_t> grid_dims{grid.shape(), grid.shape()+4}; 
  const int num_channels = grid_dims[0];
  const int grid_size = grid_dims[1];

  // The offset is the number of grid points per channel. It's used as a stride for 
  // traversing through the grid pointer

  const int offset = std::accumulate(grid_dims.begin()+1, grid_dims.end(), 1, std::multiplies<int>());
  const int num_grid_points = num_channels * offset;

  py::print("Num Channels",num_channels, "Offest ",offset);

  const float* grid_ptr = (float*)grid.request().ptr;


  npcarray estimated_coords; 
  estimate_coords(grid_ptr, num_channels, offset, estimated_coords);

  bool needs_optimize = true;
  const auto max_iterations = 1000;
  const auto threshold = 1.0e-4;
  auto iteration = 0; 

  // Set up containers for SGD

  npcarray predicted_grid = py::array_t<float>(grid_dims);  
  float* predicted_grid_ptr = (float*)predicted_grid.request().ptr;
  std::memset(predicted_grid_ptr, 0, num_grid_points);

  float grid_points_grad[num_grid_points];
  float estimated_coords_grad_ptr[estimated_coords.shape(0)*4];

  // Set up constants for grid

  float oc = INTEGRAL_NORMALIZATION_3D;
  float ic = (float)(1/(SQRT_2 * variance));

  while(iteration < max_iterations){
    // Generate the grid with the estimated points
    generate_grid_forward(estimated_coords,
                          num_channels,
                          ic,
                          oc,
                          predicted_grid_ptr,
                          grid_size);

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
                           needs_optimize)
    if (!needs_optimize){
      break;
    }
    // If the backward pass is needed, update the  
    // coordinates with the calculated gradients 

    generate_grid_backward(estimated_coords_ptr, grid_points_grad, estimated_coords_grad_ptr);
    update_coords(estimated_coords_ptr, estimated_coords_grad_ptr);
  }
  return estimated_coords;  
}

estimated_coords
const int grid_size
const int num_channels
const float ic
const float oc
float* grid

PYBIND11_MODULE(Discretizer, m) {
  m.doc() = "Retrieve atom set from grids";
  m.def("grid_to_coords", &grid_to_coords, 
        "Convert grid to a set of 4P point cloud", 
        py::arg("grid"),
        py::arg("variance") = 0.04);
}
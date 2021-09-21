#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include <cmath>
#include <iostream>
#include <random>

#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp))

#define SQRT_2 1.41421356237
#define SQRT_2OPI 0.79788456
#define INTEGRAL_NORMALIZATION_3D 0.125
#define GRAD_NORMALIZATION_3D 0.125

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
                       bool& needs_optimize){
  float total = 0;

  for (auto i = 0; i < num_grid_points; ++i){
    const auto& err = (candidate_grid[i] - original_grid[i]);
    total += err * err; 
    gradient_wrt_candidate[i] = -2*err; // Cache error
    // py::print(gradient_wrt_candidate[i]);
  }
  // py::print("Grid error: ", total);
  if (total  < threshold){
    // No need to calculate backprop as the candidate grid
    // and the original grid are close enough according to
    // the threshold set
    needs_optimize = false;
  }
}


void 
coord_to_grid_forward(const npcarray& estimated_coords,
                      const int grid_size,
                      const int num_channels,
                      const float ic,
                      const float oc,
                      float* grid){

  // Assuming cubic grids with fractional coordinates
  const float cell_w = float(1.0 / grid_size);

  // Generates a grid 
  // py::print("cell_w", cell_w, grid_size);

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

      erfx[i] = -std::erf(delta_x_lower * ic);
      erfy[i] = -std::erf(delta_y_lower * ic);
      erfz[i] = -std::erf(delta_z_lower * ic);
      // py::print(delta_y_lower);
      // py::print("error function calculations: ", i,  delta_x_lower * ic, erfx[i] );
      // py::print("error function calculations: ", i, delta_y_lower * ic, erfy[i] );
      // py::print("error function calculations: ", i, delta_z_lower * ic, erfz[i] );
      // py::print("----------------------------------------------------------------");

    }

    for (auto i = 0; i < grid_size; ++i){
      erfx[i] = erfx[i+1] - erfx[i];
      erfy[i] = erfy[i+1] - erfy[i];
      erfz[i] = erfz[i+1] - erfz[i];
      // py::print("error function calculations: ", i, erfx[i] );
      // py::print("error function calculations: ", i, erfy[i] );
      // py::print("error function calculations: ", i, erfz[i] );
      // py::print("----------------------------------------------------------------");

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
            // py::print(i, j,k , val); this is giving only 0's
          grid[id] += val; 
        }
      }
    }
  }

  // auto total = 0.0;  
  // for (auto c =0; c < num_channels; ++c){
  //   auto channel_offset = c * (stride);

  //   for(auto k = 0; k < grid_size; ++k){
  //     auto z_offset = channel_offset + k * (grid_size*grid_size);
      
  //     for(auto j = 0; j < grid_size; ++j){
  //       auto zy_offset = z_offset + j * (grid_size);
        
  //       for(auto i = 0; i < grid_size; ++i){
  //         auto id =  zy_offset + i;
  //         total += grid[id]; 
  //       }
  //     }
  //   }
  // }
  // py::print("Total for generated grid: ", total);

}


void 
generate_grid_backward(const npcarray& estimated_coords,
                       const float* grid_points_grad_wrt_points,
                       float* estimated_coords_grad_ptr,
                       const int grid_size,
                       const float variance){
  // @TO DO: I made a mistake and called the variable variance when it 
  // should be stddev. But too deep into the code to change it 
  // now. Will change during the code beautification phase - Shehtab

  // Assuming cubic grids with fractional coordinates
  const float cell_w = 1.0 / grid_size;

  const auto n_atoms = estimated_coords.shape(0);
  const auto stride = grid_size * grid_size * grid_size;

  const auto var_sq_2 = 2*variance * variance; 
  const auto grad_ic = SQRT_2OPI / variance; 

  const float ic = (float)(1/(SQRT_2 * variance));

  float* points = (float*)estimated_coords.request().ptr;

  // Zero out the gradient
  std::memset(estimated_coords_grad_ptr, 0, n_atoms * 3*sizeof(float));

  for(auto atom = 0; atom < n_atoms; ++atom){
    float erfx[grid_size+1];
    float erfy[grid_size+1];
    float erfz[grid_size+1];

    float erfx_grad[grid_size+1];
    float erfy_grad[grid_size+1];
    float erfz_grad[grid_size+1];

    auto points_offset = atom*4; 
    const auto channel = (int)points[points_offset];
    const auto x = points[points_offset+1];
    const auto y = points[points_offset+2];
    const auto z = points[points_offset+3];

    for (auto i = 0; i < grid_size+1; ++i){
      auto left_bound = cell_w * i;
      auto delta_x_lower = (left_bound-x);
      auto delta_y_lower = (left_bound-y);
      auto delta_z_lower = (left_bound-z);

      erfx[i] = std::erf(delta_x_lower * ic);
      erfy[i] = std::erf(delta_y_lower * ic);
      erfz[i] = std::erf(delta_z_lower * ic);

      const auto delta_sq_x_val = (delta_x_lower)*(delta_x_lower) /(var_sq_2);
      const auto delta_sq_y_val = (delta_y_lower)*(delta_y_lower) /(var_sq_2);
      const auto delta_sq_z_val = (delta_z_lower)*(delta_z_lower) /(var_sq_2);

      erfx_grad[i] = std::exp(-delta_sq_x_val);
      erfy_grad[i] = std::exp(-delta_sq_y_val);
      erfz_grad[i] = std::exp(-delta_sq_z_val);

      // py::print("Gaussian function calculations X: ", i, erfx_grad[i] );
      // py::print("Gaussian function calculations Y: ", i, erfy_grad[i] );
      // py::print("Gaussian function calculations Z: ", i, erfz_grad[i] );
      // py::print("----------------------------------------------------------------");

    }

    for (auto i = 0; i < grid_size; ++i){
      erfx[i] = erfx[i+1] - erfx[i];
      erfy[i] = erfy[i+1] - erfy[i];
      erfz[i] = erfz[i+1] - erfz[i];

      erfx_grad[i] =  grad_ic*(erfx_grad[i+1] - erfx_grad[i]);
      erfy_grad[i] =  grad_ic*(erfy_grad[i+1] - erfy_grad[i]);
      erfz_grad[i] =  grad_ic*(erfz_grad[i+1] - erfz_grad[i]);
      // py::print("error function calculations X: ", i, erfx_grad[i] );
      // py::print("error function calculations Y: ", i, erfy_grad[i] );
      // py::print("error function calculations Z: ", i, erfz_grad[i] );
      // py::print("----------------------------------------------------------------");
    }

    auto channel_offset = channel * (stride);

    const auto grad_points_offset = atom * 3;

    for(auto k = 0; k < grid_size; ++k){
      auto z_offset = channel_offset + k * (grid_size*grid_size);

      for(auto j = 0; j < grid_size; ++j){
        auto zy_offset = z_offset + j * (grid_size);

        for(auto i= 0; i < grid_size; ++i){

          auto& x_grad = estimated_coords_grad_ptr[grad_points_offset];
          auto& y_grad = estimated_coords_grad_ptr[grad_points_offset+1];
          auto& z_grad = estimated_coords_grad_ptr[grad_points_offset+2];

          auto id =  zy_offset + i;

          x_grad += (grid_points_grad_wrt_points[id])*GRAD_NORMALIZATION_3D*(erfx_grad[i]*erfy[j]*erfx[k]);
          y_grad += (grid_points_grad_wrt_points[id])*GRAD_NORMALIZATION_3D*(erfy_grad[j]*erfx[i]*erfz[k]);
          z_grad += (grid_points_grad_wrt_points[id])*GRAD_NORMALIZATION_3D*(erfz_grad[k]*erfx[i]*erfy[j]);  

        }
      }
    }
  }
}

void
update_coords(npcarray& estimated_coords,
              const float* estimated_coords_grad_ptr,
              const float lr){

  const auto n_atoms = estimated_coords.shape(0);
  float* points = (float*)estimated_coords.request().ptr;
  const float grad_lower_bound = 1/(-10*lr);
  const float grad_higher_bound = 1/(10*lr);

  for(auto atom = 0; atom < n_atoms; ++atom){
    auto points_offset = atom*4;
    const auto grad_points_offset = atom * 3; 
    auto& x = points[points_offset+1];
    auto& y = points[points_offset+2];
    auto& z = points[points_offset+3];

    // py::print("Coordinate Grad: ", estimated_coords_grad_ptr[grad_points_offset]);
    // py::print("Coordinate Grad: ", estimated_coords_grad_ptr[grad_points_offset+1]);
    // py::print("Coordinate Grad: ", estimated_coords_grad_ptr[grad_points_offset+2]);

    x -= lr * std::clamp(estimated_coords_grad_ptr[grad_points_offset],grad_lower_bound,grad_higher_bound);
    y -= lr * std::clamp(estimated_coords_grad_ptr[grad_points_offset+1], grad_lower_bound,grad_higher_bound);
    z -= lr * std::clamp(estimated_coords_grad_ptr[grad_points_offset+2],grad_lower_bound, grad_higher_bound);
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
  std::uniform_real_distribution<> dis(0.4, 0.6); // Generates uniform random number between 0 and 1

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



#pragma once

#include "Grid.hpp"

 
template <typename DataType, bool batched, bool weights, Device device>
class BinaryGridGenerator: public GridGeneratorBase{
  /* Define the neccesary changes here
  */

};

// Functions to be binded to Python
NPCarray BinaryERFGridGenerator(py::list molecules, 
                                  NPCArray<float> widths, 
                                  NPCArray<float> heights, 
                                  NPCArray<float> depths,
                                  NPCArray<float> variances, 
                                  int grid_size, 
                                  int num_channels){

  BinaryERFGrid<float, true, true, CPU> gridder();

  gridder.set_grid_params();
  
  return gridder.get_grid();
}

NPCarray BinaryERFGridGenerator(NPCarray<float> atoms, 
                                  float widths, 
                                  float heights, 
                                  float depths,
                                  float variances, 
                                  int grid_size, 
                                  int num_channels){

  // Check if atoms is 5D or 4D 

  // IF 5D
  BinaryERFGrid<float, false, true, CPU> gridder();
  
  // IF 4D
  // BinaryERFGrid<float, false, false, CPU> gridder();
  gridder.set_grid_params();
  
  return gridder.get_grid();
}
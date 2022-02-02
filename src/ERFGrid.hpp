#pragma once

#include "Grid.hpp"

template <typename DataType, bool batched, bool weights, Device device>
class ERFGrid : public GridGeneratorBase
{
public:
  ERFGrid();
  ~ERFGrid();
  
};

// Functions to be binded to Python
NPCarray ERFGridGenerator(py::list molecules, 
                          NPCArray<float> widths, 
                          NPCArray<float> heights, 
                          NPCArray<float> depths,
                          NPCArray<float> variances, 
                          int grid_size, 
                          int num_channels){
  // If 5D
  ERFGrid<float, true, true, CPU> gridder();
  // If 4D
  // ERFGrid<float, true, false, CPU> gridder();

  gridder.set_grid_params();
  
  return gridder.get_grid();
}

NPCarray ERFGridGenerator(NPCarray<float> atoms, 
                          float widths, 
                          float heights, 
                          float depths, 
                          float variances,
                          int grid_size, 
                          int num_channels){
  
  // If 5D
  ERFGrid<float, false, true, CPU> gridder();

  // If 4D
  // ERFGrid<float, false, false, CPU> gridder();

  gridder.set_grid_params();
  
  return gridder.get_grid();
}
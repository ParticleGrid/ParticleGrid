#pragma once

#include "../grid_generator.hpp"

template <typename T>
class Weighted_Atom : Atom{
  public:
    Weighted_Atom() = default;
    Weighted_Atom(size_t channel, T x, T y, T z, T w):Atom(channel, x, y, z){
      m_w = w;
    }
    Weighted_Atom(const Weighted_Atom& other) = default;
    Weighted_Atom& operator=(const Weighted_Atom& other) = default;
    ~Weighted_Atom() = default;

  private:   
    T m_w; 
}; 

template <typename T>
class Weighted_Molecule: public Molecule{
  Weighted_Atom<T> atoms[];
};

template <typename DataType, Device device>
class WeightedGridGenerator: public GridGeneratorBase{
  /* Define the neccesary changes here
  */

};

// Functions to be binded to the python
NPCarray WeightedERFGridGenerator(NPCarray<float> atoms, 
                                  NPCArray<float> widths, 
                                  NPCArray<float> heights, 
                                  NPCArray<float> depths, 
                                  int grid_size, 
                                  int num_channels, 
                                  NPCArray<float> variances){
  WeightedERFGrid<float, true, CPU> gridder();

  gridder.set_grid_params();
  
  return gridder.get_grid();
}

NPCarray WeightedERFGridGenerator(NPCarray<float> atoms, 
                                  float widths, 
                                  float heights, 
                                  float depths, 
                                  int grid_size, 
                                  int num_channels, 
                                  float variances){
  WeightedERFGrid<float, false, CPU> gridder();

  gridder.set_grid_params();
  
  return gridder.get_grid();
}
#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <vector>

namespace py = pybind11;

template <typename T>
using NPCarray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <bool batched, typename T> struct Container {std::vector<T> value; };
template<typename T> struct Container<false, T>{  T value; }; //Partial specialization 


template <typename T> 
class Atom {
  public:
    Atom() = default; 
    Atom(T channel, T x, T y, T z):
      m_channel(m_channel), m_x(x), m_y(y), m_z(z){}
    Atom(const Atom& other) = default;
    Atom& operator=(const Atom&) = default;
    ~Atom() = default;
  private:
    T m_channel; //  Making channel the same type quickly change
    T m_x, m_y, m_z;
};

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

template <bool weights, typename T> struct AtomType {Weighted_Atom<T> atom;};
template <false, typename T> struct AtomType { Atom<T> atom;};


template <typename T, bool weights> 
class Molecule{
  public:
    size_t m_num_atoms;
    AtomType<weights, T> m_atoms[];  
  public:
    T get_volume(){ return m_height*m_depth*m_depth;}
};

enum Device {CPU, GPU};

template <typename DataType, bool batched, bool weights, Device device>
class GridGeneratorBase {
  public: 
    std::vector<Molecule<DataType, weights>> m_molecules;
    Container<batched, DataType> variances;
    Container<batched, DataType> heights;
    Container<batched, DataType> widths;
    Container<batched, DataType> depths; 

    size_t m_num_molecules;
    size_t m_grid_size;
    size_t m_num_channels;

  public: 
    GridGeneratorBase():molecules{nullptr}; 
    GridGeneratorBase(const GridGeneratorBase& other);
    GridGeneratorBase& operator=(const GridGeneratorBase& other);
    ~GridGeneratorBase();

    template<bool is_batched = batched>
    std::enable_if_t<is_batched>
    virtual void set_grid_params(py::list molecules, 
                                 NPCarray<DataType> variances,
                                 size_t grid_size);

    template<bool is_batched = batched>
    std::enable_if_t<is_batched>
    virtual void set_grid_params(py::list molecules, 
                                 DataType variances,
                                 size_t grid_size);
    virtual NPCarray<DataType> get_grid();

  private:

    
 }; 

 // -------------------------------------------
 // Implementation
 // -------------------------------------------

template <typename DataType, true, bool weighted, Device device>
void
GridGeneratorBase::set_grid_params(py::list molecules, 
                                  NPCarray<DataType> variances,
                                  size_t grid_size){
m_num_molecules = molecules.size();

}

template <typename DataType, false, bool weighted, Device device>
void
GridGeneratorBase::set_grid_params(NPCarray<DataType> atoms, 
                                   NPCarray<DataType> variances,
                                   size_t grid_size){
  m_num_molecules = 1;

}


template <typename DataType, bool batched, bool weighted, Device device>
NPCarray<DataType> 
GridGeneratorBase::get_grid(){

  std::vector<ssize_t> grid_shape = batched ? {(ssize_t) m_num_molecules,
                                                 m_num_channels,
                                                 m_grid_size,
                                                 m_grid_size,
                                                 m_grid_size} : {(ssize_t) m_num_channels,
                                                 m_grid_size,
                                                 m_grid_size,
                                                 m_grid_size}; 
  
  auto size = m_num_molecules * m_num_channels * m_grid_size * m_grid_size * m_grid_size; 
  NPCarray<DataType> return_grid = NPCarray<DataType>(grid_shape);
  DataType* grid_ptr = (DataType*)grid.request().ptr;
  memset(grid_ptr, 0, size*sizeof(DataType));
  return  grid;
}
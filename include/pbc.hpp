#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "math_utils.hpp"
#include <string>

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> npcarray;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npcarray_int;

struct CrystalParams
{
  float m_A;
  float m_B;
  float m_C;
  float m_alpha;
  float m_beta;
  float m_gamma;

  std::vector<float> m_cart_coords;
  std::vector<int> m_channels;
  std::vector<int> m_elements;
  std::array<float, 9> m_transform_matrix;
  CrystalParams(const float &A,
                const float &B,
                const float &C,
                const float &alpha,
                const float &beta,
                const float &gamma,
                const npcarray &coords,
                const npcarray_int &elements) : m_A(A), m_B(B), m_C(C),
                                                m_alpha(alpha), m_beta(beta), m_gamma(gamma),
                                                m_transform_matrix(Frac2CarTransformMatrix(A, B, C, alpha, beta, gamma))
  {
    add_expanded_coords(coords, elements);
  }

  ~CrystalParams() = default;

  std::string
  toString() const
  {
    return std::string("Lattice parameters: \n") +
           std::string("A:\t") + std::to_string(m_A) + "\n" +
           std::string("B:\t") + std::to_string(m_B) + "\n" +
           std::string("C:\t") + std::to_string(m_C) + " \n" +
           std::string("Alpha:\t") + std::to_string(m_alpha) + "\n" +
           std::string("Beta:\t") + std::to_string(m_beta) + "\n" +
           std::string("Gamma:\t") + std::to_string(m_gamma) + "\n";
  }

  void
  add_expanded_coords(const npcarray &coords, const npcarray_int &elements)
  {
    const size_t num_points = coords.shape(0);
    const size_t num_dims = coords.shape(1);

    if (num_dims != 4)
    {
      throw std::runtime_error("Number of dimensions must be 4");
    }

    const float *ptr_points = (float *)coords.request().ptr;
    const int *elem_data = (int *)elements.request().ptr;

    std::vector<float> expanded_frac_coords;

    for (size_t coord = 0; coord < num_points; ++coord)
    {
      // For each fractional coordinate in the unit cell
      const float x = ptr_points[coord * 4];
      const float y = ptr_points[coord * 4 + 1];
      const float z = ptr_points[coord * 4 + 2];
      const float c = ptr_points[coord * 4 + 3];
      const int &elem = elem_data[coord];

      expanded_frac_coords.insert(expanded_frac_coords.end(), {x, y, z});
      m_channels.push_back(int(c));
      m_elements.push_back(elem);

      for (const auto &x_translate : {x - 1, x, x + 1})
      {
        // Translate along the x-dimension
        if (x_translate > -0.25 && x_translate < 1.25)
        {
          for (const auto &y_translate : {y - 1, y, y + 1})
          {
            // Translate along the y-dimension
            if (y_translate > -0.25 && y_translate < 1.25)
            {
              for (const auto &z_translate : {z - 1, z, z + 1})
              {
                // Translate along the z-dimension
                if (z_translate > -0.25 && z_translate < 1.25)
                {
                  if (x_translate == x && y_translate == y && z_translate == z)
                  {
                    continue;
                  }
                  expanded_frac_coords.insert(expanded_frac_coords.end(), {x_translate, y_translate, z_translate});
                  m_channels.push_back(c);
                  m_elements.push_back(elem);
                }
              }
            }
          }
        }
      }
    }

    std::vector<float> expanded_cart_coords(expanded_frac_coords.size(), 0.0);

    matmul<float>(expanded_frac_coords.data(),
                  m_transform_matrix.data(),
                  expanded_cart_coords.data(),
                  expanded_cart_coords.size() / 3);
    m_cart_coords = std::move(expanded_cart_coords);
  }
  py::array_t<float> LJ_grid(const int &grid_size);
  py::array_t<float> Metal_Organic_Grid(const int &grid_size, const float &variance);
  py::array_t<float> get_cart_coords();
  py::array_t<int> get_elements();
  py::array_t<int> get_channels();
  py::array_t<float> get_transform_matrix();
};

#pragma once
#include <stddef.h>
#include <array>
#include <vector>
#include <iostream>
#include <cmath>

template <typename T>
inline void
matmul(const T *input_matrix,
       const std::array<T, 9> &transform,
       T *output_matrix,
       const size_t &m)
{

  // Make inline as it may be used in multiple source files
  constexpr size_t k = 3;

#pragma omp parallel for
  for (auto row = 0; row < m; ++row)
  {
    for (auto col = 0; col < k; ++col)
    {
      T &y_ij = output_matrix[row * k + col];

      for (auto i = 0; i < 3; ++i)
      {
        const auto &x_ij = input_matrix[row * k + i];
        const auto &w_ab = transform[i * k + col];
        y_ij += x_ij * w_ab;
      }
    }
  }
}

template <typename T>
inline void matmul(const T *input_matrix,
                   const T *transform,
                   T *output_matrix,
                   const size_t &m)
{
  constexpr size_t k = 3;
#pragma omp parallel for
  for (auto row = 0; row < m; ++row)
  {
    for (auto col = 0; col < k; ++col)
    {
      T &y_ij = output_matrix[row * k + col];

      for (auto i = 0; i < 3; ++i)
      {
        const auto &x_ij = input_matrix[row * k + i];
        const auto &w_ab = transform[i * k + col];
        y_ij += x_ij * w_ab;
      }
    }
  }
}

template <typename T>
inline void MeshGrid_Cartesian_3D(const int &grid_size,
                                  T *output_matrix)
{
  // Linearized meshgrid in 3D space
  const int num_meshgrid_points = 3 * grid_size * grid_size * grid_size;
  std::vector<T> input_points(num_meshgrid_points);
  const float delta = 1.0 / grid_size;
  for (auto meshgrid_point = 0; meshgrid_point < num_meshgrid_points; ++meshgrid_point)
  {
    auto x_frac_coords = (meshgrid_point / (grid_size * grid_size));
    auto y_frac_coords = (x_frac_coords * grid_size ) % grid_size;
    auto z_frac_coords = meshgrid_point % grid_size;

    input_points.push_back(float(x_frac_coords));
    input_points.push_back(float(y_frac_coords));
    input_points.push_back(float(z_frac_coords));
  }
}

template <typename T>
inline std::array<T, 9> Frac2CarTransformMatrix(const T &a,
                                                const T &b,
                                                const T &c,
                                                const T &alpha,
                                                const T &beta,
                                                const T &gamma)
{
  std::array<T, 9> _transform_matrix;
  /*
  [a            0                 0                           ]
  [b*cos(gamma) b*sin(gamma)      0                           ]
  [c*cos(beta)  c*omega           c*sqrt(sin^2(beta)-omega^2) ]


  omega = (cos(alpha)-(cos(gamma)*cos(beta))) / sin(gamma)

  */

  T omega = (std::cos(alpha) - std::cos(gamma) * std::cos(beta)) / (std::sin(gamma));

  _transform_matrix[0] = a;
  _transform_matrix[1] = T(0);
  _transform_matrix[2] = T(0);

  _transform_matrix[3] = b * std::cos(gamma);
  _transform_matrix[4] = b * std::sin(gamma);
  _transform_matrix[5] = T(0);

  _transform_matrix[6] = c * std::cos(beta);
  _transform_matrix[7] = c * omega;
  _transform_matrix[8] = c * std::sqrt(std::pow(std::sin(beta), 2) - std::pow(omega, 2));

  return _transform_matrix;
}

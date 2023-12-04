#include "pbc.hpp"
#include "uff.hpp"
#include "avx_erf.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> npcarray;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npcarray_int;

// py::array_t<float>
// energy_grid(const CrystalParams &crystal_params,
//             const ssize_t &W,
//             const ssize_t &H,
//             const ssize_t &D)
// {
//   const std::array<ssize_t, 4> grid_shape = {1, H, W, D};
//   npcarray grid = py::array_t<float>(grid_shape);

//   float* tensor = (float *)grid.request().ptr;

//   // periodic::generate_grid_impl<>(tensor);
//   return grid;
// }

// py::array_t<float>
// periodic_grid(const CrystalParams &crystal_params,
//               const ssize_t &W,
//               const ssize_t &H,
//               const ssize_t &D,
//               const ssize_t &num_channels)
// {
//   const std::array<ssize_t, 4> grid_shape = {num_channels, H, W, D};
//   npcarray grid = py::array_t<float>(grid_shape);

//   float* tensor = (float *)grid.request().ptr;

//   periodic::generate_grid_impl<>(tensor);
//   return grid;
// }


py::array_t<float>
CrystalParams::LJ_grid(const int &grid_size)
{
  const std::array<ssize_t, 4> grid_shape = {1, grid_size, grid_size, grid_size};
  npcarray grid = py::array_t<float>(grid_shape);

  float delta = 1.0 / grid_size;

  float *tensor = (float *)grid.request().ptr;

  for (auto x_i = 0; x_i < grid_size; ++x_i)
  {
    const float gp_x = (x_i + 0.5) * delta;
    for (auto y_i = 0; y_i < grid_size; ++y_i)
    {
      const float gp_y = (y_i + 0.5) * delta;
      const int stride = x_i * (grid_size * grid_size) + y_i * (grid_size);

      for (auto z_i = 0; z_i < grid_size; ++z_i)
      {
        const float gp_z = (z_i + 0.5) * delta;

        float frac_coords[3] = {gp_x, gp_y, gp_z};

        float cart_coords[3];

        matmul<float>(frac_coords,
                      this->m_transform_matrix.data(),
                      cart_coords,
                      1);
        float energy = 0.0;
        for (auto atom = 0; atom < m_cart_coords.size() / 3; ++atom)
        {
          float atom_x = this->m_cart_coords[3 * atom];
          float atom_y = this->m_cart_coords[3 * atom + 1];
          float atom_z = this->m_cart_coords[3 * atom + 2];

          float r = std::sqrt(std::pow((atom_x - gp_x), 2) +
                              std::pow((atom_y - gp_y), 2) +
                              std::pow((atom_z - gp_z), 2));

          float sigma = 0.5 * (SIGMA_ARRAY[0] + SIGMA_ARRAY[m_elements[atom]]);
          energy += (std::pow(sigma / r, 12) - std::pow(sigma / r, 6));
        }

        if (energy < 1E8)
        {
          tensor[stride + z_i] += 1 / (4 * energy);
        }
      }
    }
  }
  return grid;
}


py::array_t<float>
CrystalParams::Metal_Organic_Grid(const int &grid_size, const float &variance)
{
  const std::array<ssize_t, 4> grid_shape = {2, grid_size, grid_size, grid_size};
  npcarray grid = py::array_t<float>(grid_shape);

  const float delta = 1.0 / grid_size;

  float *tensor = (float *)grid.request().ptr;

  constexpr float outer_const = 0.125;
  constexpr float inner_const = 2.82842712 / 8;
  const float var_div = std::pow(variance, 3);
  
  // I believe I know a better way of doing by performing all calculations
  // in the fractional space using a metric tensor
  // Follow up with Shehtab ( or me in case of future self)
  // - Shehtab

  const int channel_stride = grid_size * grid_size * grid_size;

  float gp_x, gp_x_delta = 0;
  float gp_y, gp_y_delta = 0;
  float gp_z, gp_z_delta = 0;

  float frac_coords_xyz_delta[3] = {gp_x, gp_y, gp_z};
  float cart_coords_xyz_delta[3];

  matmul<float>(frac_coords_xyz_delta,
                this->m_transform_matrix.data(),
                cart_coords_xyz_delta,
                1);

  float cart_coords_xyz[3] = {0.0, 0.0, 0.0};

  for (auto x = 0; x < grid_size; ++x)
  {
    gp_x = gp_x_delta;
    gp_x_delta = (x + 1) / delta;
    gp_y_delta = 0.0;
    for (auto y = 0; y < grid_size; ++y)
    {
      gp_y = gp_y_delta;
      gp_y_delta = (y + 1) / delta;
      gp_z_delta = 0.0;
      const int stride = x * (grid_size * grid_size) + y * (grid_size);
      for (auto z = 0; z < grid_size; ++z)
      {
        gp_z = gp_z_delta;
        gp_z_delta = (z + 1) / delta;

        cart_coords_xyz[0] = cart_coords_xyz_delta[0];
        cart_coords_xyz[1] = cart_coords_xyz_delta[1];
        cart_coords_xyz[2] = cart_coords_xyz_delta[2];

        float temp[3] = {gp_x_delta, gp_y_delta, gp_z_delta};

        matmul<float>(temp,
                      this->m_transform_matrix.data(),
                      cart_coords_xyz_delta,
                      1);
        for (auto atom = 0; atom < m_cart_coords.size() / 3; ++atom)
        {
          const auto &atom_channel = m_channels[atom] * channel_stride;
          float atom_x = this->m_cart_coords[3 * atom];
          float atom_y = this->m_cart_coords[3 * atom + 1];
          float atom_z = this->m_cart_coords[3 * atom + 2];

          float l_end_point = erf_apx(gp_x - atom_x) * erf_apx(gp_y - atom_y) * erf_apx(gp_z - atom_z);
          float r_end_point = erf_apx(gp_x_delta - atom_x) * erf_apx(gp_y_delta - atom_y) * erf_apx(gp_z_delta - atom_z);
          
          float prob = r_end_point - l_end_point;
          if (prob > 1e-7){
            tensor[atom_channel + stride + z] += ((prob * inner_const) / var_div);
          }
        }
      }
    }
  }
  return grid;
}

PYBIND11_MODULE(Periodic, m)
{
  m.doc() = "Grid generator for periodic crystals";
  py::class_<CrystalParams>(m, "CrystalParams")
      .def(py::init<const float &,
                    const float &,
                    const float &,
                    const float &,
                    const float &,
                    const float &,
                    const npcarray &,
                    const npcarray_int &>(),
           py::kw_only(), py::arg("A"), py::arg("B"), py::arg("C"),
           py::arg("alpha"), py::arg("beta"), py::arg("gamma"), py::arg("coords"), py::arg("elements"))
      .def("__repr__", &CrystalParams::toString)
      .def("LJ_Grid", &CrystalParams::LJ_grid)
      .def("Probability_Grid", &CrystalParams::Metal_Organic_Grid);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
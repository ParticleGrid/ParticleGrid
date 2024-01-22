#include "pbc.hpp"
#include "uff.hpp"
#include "avx_erf.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> npcarray;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npcarray_int;

py::array_t<float>
CrystalParams::LJ_grid(const size_t &grid_size)
{
  const std::array<ssize_t, 4> grid_shape = {1, (ssize_t)grid_size, (ssize_t)grid_size, (ssize_t)grid_size};
  npcarray grid = py::array_t<float>(grid_shape);
  float delta = 1.0 / grid_size;
  float *tensor = (float *)grid.request().ptr;
  memset(tensor, 0, grid_size * grid_size * grid_size * sizeof(float));
  // std::cout << "Grid size: " << grid_size << " Delta: " << delta << "\n";
  // std::cout << "Starting LJ grid calculation\n";
#pragma omp parallel for
  for (size_t x_i = 0; x_i < grid_size; ++x_i)
  {
    const float gp_x = (x_i + 0.5) * delta;
    for (size_t y_i = 0; y_i < grid_size; ++y_i)
    {
      const float gp_y = (y_i + 0.5) * delta;
      const int stride = x_i * (grid_size * grid_size) + y_i * (grid_size);
      for (size_t z_i = 0; z_i < grid_size; ++z_i)
      {
        const float gp_z = (z_i + 0.5) * delta;
        float frac_coords[3] = {gp_x, gp_y, gp_z};
        float cart_coords[3] = {0.0, 0.0, 0.0};
        matmul<float>(frac_coords,
                      this->m_transform_matrix.data(),
                      cart_coords,
                      1);
        const float grid_x = cart_coords[0];
        const float grid_y = cart_coords[1];
        const float grid_z = cart_coords[2];
        // if (x_i == 4 && y_i == 0 && z_i == 5)
        // {
        //   std::cout << "Grid coords: " << grid_x
        //             << ", " << grid_y << ", " << grid_z << std::endl;
        // }
        float energy = 0.0;
        for (size_t atom = 0; atom < m_cart_coords.size() / 3; ++atom)
        {
          float atom_x = this->m_cart_coords[3 * atom];
          float atom_y = this->m_cart_coords[3 * atom + 1];
          float atom_z = this->m_cart_coords[3 * atom + 2];
          float r = std::sqrt(std::pow((atom_x - grid_x), 2) +
                              std::pow((atom_y - grid_y), 2) +
                              std::pow((atom_z - grid_z), 2));

          float sigma = 0.5 * (SIGMA_ARRAY[0] + SIGMA_ARRAY[m_elements[atom]]);
          float local_energy = (std::pow(sigma / r, 12) - std::pow(sigma / r, 6));
          energy += local_energy;
          // if (x_i == 4 && y_i == 0 && z_i == 5)
          // {

          //   std::cout << "Atom coords: " << atom_x
          //             << ", " << atom_y << ", " << atom_z
          //             << ", " << r << ", " << local_energy << std::endl;
          // }
        }
        if (energy < 1E8)
        {
<<<<<<< HEAD
          tensor[stride + z_i] = 1 / (4 * energy);
=======
          tensor[stride + z_i] += 4 * energy;
>>>>>>> Force function to return raw scores
        }
        // if (x_i == 4 && y_i == 0 && z_i == 5)
        // {
        //   std::cout << energy << ", " << tensor[stride + z_i] << std::endl;
        // }
      }
    }
  }
  // std::cout << "Finished LJ grid calculation\n";
  return grid;
}

// TO DO: ADD specialized implementation for cached grid values and orthogonal lattices
// - Shehtab
py::array_t<float>
CrystalParams::Metal_Organic_Grid(const size_t &grid_size, const float &variance)
{
  const std::array<ssize_t, 4> grid_shape = {2, (ssize_t)grid_size, (ssize_t)grid_size, (ssize_t)grid_size};
  npcarray grid = py::array_t<float>(grid_shape);
  const float delta = 1.0 / (grid_size);
  float *tensor = (float *)grid.request().ptr;
  memset(tensor, 0, grid_size * grid_size * grid_size * 2 * sizeof(float));
  constexpr float outer_const = 0.125;
  const float &inner_const = float(0.707106781) / variance;

  // I believe I know a better way of doing by performing all calculations
  // in the fractional space using a metric tensor
  // Follow up with Shehtab ( or me in case of future self)
  // - Shehtab

  const int channel_stride = grid_size * grid_size * grid_size;

#pragma omp parallel for
  for (size_t x = 0; x < grid_size; ++x)
  {
    float gp_x = x * delta;
    float gp_x_delta = (x + 1) * delta;
    for (size_t y = 0; y < grid_size; ++y)
    {
      float gp_y = y * delta;
      float gp_y_delta = (y + 1) * delta;
      const int stride = x * (grid_size * grid_size) + y * (grid_size);
      for (size_t z = 0; z < grid_size; ++z)
      {
        float gp_z = z * delta;
        float gp_z_delta = (z + 1) * delta;
        float temp[3] = {gp_x, gp_y, gp_z};
        float temp_xyz[3] = {0.0, 0.0, 0.0};
        matmul<float>(temp,
                      this->m_transform_matrix.data(),
                      temp_xyz,
                      1);

        const float grid_x = temp_xyz[0];
        const float grid_y = temp_xyz[1];
        const float grid_z = temp_xyz[2];

        float temp_delta[3] = {gp_x_delta, gp_y_delta, gp_z_delta};
        float temp_xyz_delta[3] = {0.0, 0.0, 0.0};
        matmul<float>(temp_delta,
                      this->m_transform_matrix.data(),
                      temp_xyz_delta,
                      1);
        const float grid_x_delta = temp_xyz_delta[0];
        const float grid_y_delta = temp_xyz_delta[1];
        const float grid_z_delta = temp_xyz_delta[2];

        for (size_t atom = 0; atom < m_cart_coords.size() / 3; ++atom)
        {
          const auto &atom_channel = m_channels[atom] * channel_stride;
          const float &atom_x = this->m_cart_coords[3 * atom];
          const float &atom_y = this->m_cart_coords[3 * atom + 1];
          const float &atom_z = this->m_cart_coords[3 * atom + 2];

          const float &l_x = inner_const * (grid_x_delta - atom_x);
          const float &l_y = inner_const * (grid_y_delta - atom_y);
          const float &l_z = inner_const * (grid_z_delta - atom_z);

          const float &r_x = inner_const * (grid_x - atom_x);
          const float &r_y = inner_const * (grid_y - atom_y);
          const float &r_z = inner_const * (grid_z - atom_z);

          float integral_x = erf_apx(l_x) - erf_apx(r_x);
          float integral_y = erf_apx(l_y) - erf_apx(r_y);
          float integral_z = erf_apx(l_z) - erf_apx(r_z);

          float prob = integral_x * integral_y * integral_z;

          if (prob > 1e-7)
          {
            tensor[atom_channel + stride + z] += (prob * outer_const);
          }
        }
      }
    }
  }
  return grid;
}

py::array_t<float>
CrystalParams::get_cart_coords()
{
  const std::array<ssize_t, 2> grid_shape = {(ssize_t)(m_cart_coords.size() / 3), 4};
  npcarray grid = py::array_t<float>(grid_shape);

  float *tensor = (float *)grid.request().ptr;

  for (size_t atom = 0; atom < m_cart_coords.size() / 3; ++atom)
  {
    tensor[atom * 4] = m_cart_coords[3 * atom];
    tensor[atom * 4 + 1] = m_cart_coords[3 * atom + 1];
    tensor[atom * 4 + 2] = m_cart_coords[3 * atom + 2];
    tensor[atom * 4 + 3] = m_elements[atom];
  }
  return grid;
}

py::array_t<int>
CrystalParams::get_elements()
{
  const std::array<ssize_t, 1> grid_shape = {(ssize_t)m_elements.size()};
  npcarray_int grid = py::array_t<int>(grid_shape);

  int *tensor = (int *)grid.request().ptr;

  for (size_t atom = 0; atom < m_elements.size(); ++atom)
  {
    tensor[atom] = m_elements[atom];
  }
  return grid;
}

py::array_t<int>
CrystalParams::get_channels()
{
  const std::array<ssize_t, 1> grid_shape = {(ssize_t)m_channels.size()};
  npcarray_int grid = py::array_t<int>(grid_shape);

  int *tensor = (int *)grid.request().ptr;

  for (size_t atom = 0; atom < m_channels.size(); ++atom)
  {
    tensor[atom] = m_channels[atom];
  }
  return grid;
}

py::array_t<float>
CrystalParams::get_transform_matrix()
{
  const std::array<ssize_t, 2> grid_shape = {3, 3};
  npcarray grid = py::array_t<float>(grid_shape);

  float *tensor = (float *)grid.request().ptr;

  for (auto i = 0; i < 3; ++i)
  {
    for (auto j = 0; j < 3; ++j)
    {
      tensor[i * 3 + j] = m_transform_matrix[i * 3 + j];
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
      .def("Probability_Grid", &CrystalParams::Metal_Organic_Grid)
      .def("get_cartesian_coords", &CrystalParams::get_cart_coords)
      .def("get_transform_matrix", &CrystalParams::get_transform_matrix)
      .def("get_elements", &CrystalParams::get_elements)
      .def("get_channels", &CrystalParams::get_channels);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
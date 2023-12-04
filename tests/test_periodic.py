import numpy as np


def test_import():
  from Periodic import CrystalParams


def test_crystal_param():
  from Periodic import CrystalParams

  coords = np.array([[0, 0, 0, 0], [0, 0.5, 0.5, 0.5]])
  elements = np.array([0, 0])
  params = CrystalParams(
        A=10,
        B=10,
        C=10,
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        coords=coords,
        elements=elements,
    )
  print(params)
  print(dir(params))

  energy_grid = params.LJ_Grid(32);
  print(type(energy_grid))
  print(energy_grid.shape)
  print(energy_grid[0][0])


if __name__ == "__main__":
  test_import()
  test_crystal_param()

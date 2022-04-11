from statistics import mean, stdev
from time import perf_counter
import numpy as np
from scipy.spatial import distance
from scipy import special
from tqdm import tqdm


def NPGridGenerator(mol, grid_size, channels, variance):
  x_max, y_max, z_max = mol[:, 1:].max(axis=0)
  x_coords = np.linspace(0., x_max, grid_size + 1, dtype=np.float32)
  y_coords = np.linspace(0., y_max, grid_size + 1, dtype=np.float32)
  z_coords = np.linspace(0., z_max, grid_size + 1, dtype=np.float32)

  x_a, y_a, z_a = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

  x_a = distance.cdist(x_a.reshape(-1, 1),
                       mol[:, 1].reshape(-1, 1),
                       lambda u, v: v - u)  # type: ignore
  y_a = distance.cdist(y_a.reshape(-1, 1),
                       mol[:, 2].reshape(-1, 1),
                       lambda u, v: v - u)  # type: ignore
  z_a = distance.cdist(z_a.reshape(-1, 1),
                       mol[:, 3].reshape(-1, 1),
                       lambda u, v: v - u)  # type: ignore

  u_sub = np.sqrt(2) / (2 * variance)

  err_x = special.erf(u_sub * x_a).reshape(grid_size + 1,
                                           grid_size + 1,
                                           grid_size + 1,
                                           -1)
  err_y = special.erf(u_sub * y_a).reshape(grid_size + 1,
                                           grid_size + 1,
                                           grid_size + 1,
                                           -1)
  err_z = special.erf(u_sub * z_a).reshape(grid_size + 1,
                                           grid_size + 1,
                                           grid_size + 1,
                                           -1)
  err_x = np.swapaxes(err_x, 0, 3)
  err_y = np.swapaxes(err_y, 0, 3)
  err_z = np.swapaxes(err_z, 0, 3)

  x_vals = err_x[:, : grid_size, : grid_size, : grid_size] - err_x[:, 1: , 1: , 1:]
  y_vals = err_y[:, : grid_size, : grid_size, : grid_size] - err_y[:, 1: , 1: , 1:]
  z_vals = err_z[:, : grid_size, : grid_size, : grid_size] - err_z[:, 1: , 1: , 1:]

  out = np.multiply(np.multiply(x_vals, y_vals), z_vals) / 8
  
  generated_grid = np.zeros((channels, grid_size, grid_size, grid_size))

  for i in range(channels):
    inds = mol[:, 0] == i
    if (inds.any() == True):
      generated_grid[i] = out[inds].sum(0).reshape(grid_size, grid_size, grid_size)
  return generated_grid  


if __name__ == '__main__':
  times = []  
  for i in tqdm(range(10)):
    coords = np.random.randn(20,3) * 10
    channels = np.random.randint(0, 8, size=(20,1))
    mol = np.hstack((channels, coords))

    timer_start = perf_counter()
    test = NPGridGenerator(mol, 32, 8, 0.25)
    timer_end = perf_counter()
    times.append(timer_end-timer_start)
  print(f"Average time (s): {mean(times)}  standard deviation {stdev(times)}")

from statistics import mean, stdev
from time import perf_counter
import numpy as np
from numba import jit, vectorize, guvectorize, float32, int64
from scipy import special
from tqdm import tqdm
from numba.np.unsafe.ndarray import to_fixed_tuple, tuple_setitem


@vectorize()
def diff(x, y):
  return y - x

@vectorize([float32(float32)])
def erf(x):
  return special.erf(x)

@jit(nopython=True)
def meshgrid(x, y, z):
    """ Taken from: 
        https://stackoverflow.com/questions/70613681/numba-compatible-numpy-meshgrid
    """
    xx = np.empty(shape=(x.size, y.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(x.size, y.size, z.size), dtype=z.dtype)
    for i in range(z.size):
        for j in range(y.size):
            for k in range(x.size):
                xx[i,j,k] = k  # change to x[k] if indexing xy
                yy[i,j,k] = j  # change to y[j] if indexing xy
                zz[i,j,k] = i  # change to z[i] if indexing xy
    return zz, yy, xx

@jit()
def cdist(grid_points, coords):
  ret = np.empty(shape=(grid_points.size, coords.size), dtype=np.float32)
  for i in range(grid_points.size):
    for j in range(coords.size):
      ret[i][j] = diff(grid_points[i], coords[j])
  return ret

@jit([float32[:,:,:,:](float32[:,:,:,:], int64, int64)])
def swapaxes(a, axis1, axis2):
    """ Numba swapaxes ala np.swapaxes() """
    axis1 = a.ndim + axis1 if axis1 < 0 else axis1
    axis2 = a.ndim + axis2 if axis2 < 0 else axis2
    if axis1 == axis2:
        return a
    pos = to_fixed_tuple(np.arange(a.ndim), a.ndim)
    tmp = pos[axis1]
    pos = tuple_setitem(pos, axis1, pos[axis2])
    pos = tuple_setitem(pos, axis2, tmp)
    return a.transpose(pos)

@jit()
def NUMBAGridGenerator(mol, grid_size, channels, variance):
  x_max = mol[:, 0].max()
  y_max = mol[:, 1].max()
  z_max = mol[:, 2].max()

  x_coords = np.linspace(np.float32(0.0),
                         x_max,
                         grid_size + 1)
  y_coords = np.linspace(np.float32(0.0),
                         y_max,
                         grid_size + 1)
  z_coords = np.linspace(np.float32(0.0),
                         z_max,
                         grid_size + 1)

  x_a, y_a, z_a = meshgrid(x_coords, y_coords, z_coords)

  x_a = cdist(x_a.reshape(-1, 1),
                       mol[:, 1])  # type: ignore
  y_a = cdist(y_a.reshape(-1, 1),
                       mol[:, 2])  # type: ignore
  z_a = cdist(z_a.reshape(-1, 1),
                       mol[:, 3])  # type: ignore

  u_sub = np.float32(np.sqrt(2) / (2 * variance))

  err_x = erf(u_sub * x_a).reshape(grid_size + 1,
                                           grid_size + 1,
                                           grid_size + 1,
                                           -1)
  err_y = erf(u_sub * y_a).reshape(grid_size + 1,
                                   grid_size + 1,
                                   grid_size + 1,
                                  -1)
  err_z = erf(u_sub * z_a).reshape(grid_size + 1,
                                   grid_size + 1,
                                   grid_size + 1,
                                   -1)
  err_x = np.swapaxes(err_x, np.int64(0), np.int64(3))
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
    mol = np.hstack((channels, coords)).astype(np.float32)
    print(mol.shape)
    timer_start = perf_counter()
    test = NUMBAGridGenerator(mol, 32, 8, 0.25)
    timer_end = perf_counter()
    times.append(timer_end-timer_start)
  print(f"Average time (s): {mean(times)}  standard deviation {stdev(times)}")
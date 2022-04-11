from copyreg import pickle
from time import perf_counter
from NUMPY_Implementation import NPGridGenerator
from NUMBA_Implementation import NUMBAGridGenerator
import GridGenerator as gg


def main():
  grid_sizes = [16, 32, 64, 128]
  variances = [0.1, 0.25, 0.5, 1]  # This controls how dense the grids are
  generators = [(NPGridGenerator, "Numpy"),
                (NUMBAGridGenerator, "Numba"),
                (gg.molecule_grid, "ParticleGrid-Sparse")]
  num_trials = 10
  GridGenTimes = {}

  mol_list = pickle.load(open("molecule_list.pickle", 'r'))
  for gridder, title in generators:
    for gsize in grid_sizes:
      for variance in variances:
        _times = []
        for trials in range(num_trials):
          start = perf_counter()
          for molecule in mol_list:
            tensor = gridder(molecule, gsize, 8, variance)
          elapsed_time = perf_counter() - start
          _times.append(elapsed_time)
        GridGenTimes[title][gsize][variance] = _times

  # To do: Do some analysis with the benchmarks or save 
  #        them at least
  #


if __name__ == '__main__':
  main()

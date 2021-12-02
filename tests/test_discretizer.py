def test_import():
    from ParticleGrid import grid_to_coords


def test_discretizer_multi_atom_multi_channel():
    from ParticleGrid import coord_to_grid
    from Discretizer import optimize_coords
    import numpy as np

    test_point = np.array([[0, 0.5, 0.5, 0.5],
                           [1, 0.5, 0.5, 0.5]])

    grid = coord_to_grid(test_point,
                         width=1,
                         height=1,
                         depth=1,
                         num_channels=2,
                         grid_size=32,
                         variance=0.05)

    some_noise = np.random.rand(2,3) * (0.1)


    noisy_approximation = test_point.copy()
    noisy_approximation[:,1:] = noisy_approximation[:,1:]  + some_noise

    print("Noisy Approximation: \n",noisy_approximation)

    estimated_coords = optimize_coords(grid,
                                       noisy_approximation,
                                       variance=0.05, 
                                       tolerance=3e-7)

    print("Optimized coordinations: \n",estimated_coords, "\n",test_point)
    assert np.allclose(test_point, estimated_coords)


if __name__ == '__main__':
    test_discretizer_multi_atom_multi_channel()

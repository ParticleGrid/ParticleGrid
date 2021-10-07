def test_import():
    from ParticleGrid import grid_to_coords


def test_discretizer_multi_atom_multi_channel():
    from ParticleGrid import coord_to_grid
    from Discretizer import grid_to_coords
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
    estimated_coords = grid_to_coords(grid)
    print(estimated_coords, "\n",test_point)
    assert np.allclose(test_point, estimated_coords)

def test_discretizer_multi_atom_single_channel():
    from ParticleGrid import coord_to_grid
    from Discretizer import grid_to_coords
    import numpy as np

    test_points = np.array([[0, 0.5, 0.5, 0.5],
                            [0, 0.25, 0.25, 0.25]])

    grid = coord_to_grid(test_points,
                         width=1,
                         height=1,
                         depth=1,
                         num_channels=1,
                         grid_size=32,
                         variance=0.05)
    print(np.sum(grid))
    estimated_coords = grid_to_coords(grid)
    print(estimated_coords, "\n", test_points)
    assert np.allclose(test_points, estimated_coords) 

def test_discretizer_single_atom_single_channel():
    from ParticleGrid import coord_to_grid
    from Discretizer import grid_to_coords
    import numpy as np

    test_points = np.array([[0, 0.5, 0.5, 0.5]])

    grid = coord_to_grid(test_points,
                         width=1,
                         height=1,
                         depth=1,
                         num_channels=1,
                         grid_size=32,
                         variance=0.05)
    print("Sum of original grid: ", np.sum(grid))
    estimated_coords = grid_to_coords(grid, 0.05, tolerance=1e-7, max_iterations=10000)
    print(estimated_coords, "\n", test_points)
    assert np.allclose(test_points, estimated_coords, rtol=1e-04, atol=1e-6) 


if __name__ == '__main__':
    test_discretizer_single_atom_single_channel()

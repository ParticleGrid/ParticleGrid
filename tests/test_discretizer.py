def test_import():
    from ParticleGrid import grid_to_coords


def test_discretizer():
    from ParticleGrid import coord_to_grid
    from Discretizer import grid_to_coords
    import numpy as np

    center_point = np.array([[0, 0, 0, 0]])

    grid = coord_to_grid(center_point,
                         width=1,
                         height=1,
                         depth=1,
                         num_channels=2,
                         grid_size=32,
                         variance=0.05)
    print(np.sum(grid))
    estimated_coords = grid_to_coords(grid)
    print(estimated_coords, center_point)
    assert np.allclose(center_point, estimated_coords) 


if __name__ == '__main__':
    test_discretizer()

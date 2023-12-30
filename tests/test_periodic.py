import numpy as np


def generate_LJ_grid(cartesian_coords, elements, transform_matrix, grid_size=16):
    """
    Generate LJ grid for a given crystal structure

    Parameters
    ----------
    cartesian_coords : np.ndarray
        Cartesian coordinates of the crystal structure
    elements : np.ndarray
        Elements of the crystal structure
    transform_matrix : np.ndarray
        Transformation matrix of the crystal structure

    Returns
    -------
    energy_grid : np.ndarray
        LJ grid of the crystal structure
    """

    grid_points = np.linspace(0, 1, grid_size + 1)
    cart_grid_x, cart_grid_y, cart_grid_z = np.meshgrid(
        grid_points, grid_points, grid_points, indexing="ij"
    )

    frac_grid = np.concatenate(
        [
            cart_grid_x.reshape(-1, 1),
            cart_grid_y.reshape(-1, 1),
            cart_grid_z.reshape(-1, 1),
        ],
        axis=1,
    )
    hydrogen_sigma = 0.2571133701e01
    sigma_array = [0.2571133701e01, 0.2104302772e01]

    cart_grid = np.matmul(frac_grid, transform_matrix).reshape(
        grid_size + 1, grid_size + 1, grid_size + 1, 3
    )
    # print(cart_grid)

    lj_grid = np.zeros((grid_size, grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                gp_x = (cart_grid[i][j][k][0] + cart_grid[i + 1][j][k][0]) / 2
                gp_y = (cart_grid[i][j][k][1] + cart_grid[i][j + 1][k][1]) / 2
                gp_z = (cart_grid[i][j][k][2] + cart_grid[i][j][k + 1][2]) / 2
                # if (i == 4) and (j == 0) and (k == 5):
                #     print("Python Cartesian Coords: ", gp_x, gp_y, gp_z)
                energy = 0
                for atom_i, atom_coord in enumerate(cartesian_coords):
                    diff = (atom_coord[:3] - np.array([gp_x, gp_y, gp_z])) ** 2
                    dist = np.sqrt(np.sum(diff))
                    sigma = 0.5 * (hydrogen_sigma + sigma_array[elements[atom_i]])
                    local_energy = (sigma / dist) ** 12 - (sigma / dist) ** 6
                    # if (i == 4) and (j == 0) and (k == 5):
                    #     print(
                    #         "Python Atom Coords: ", atom_coord[:3], dist, local_energy
                    #     )
                    # print("Python Distance: ", dist)
                    energy += local_energy
                if energy < 1e8:
                    # print(energy, i, j, k)
                    lj_grid[i][j][k] += 1 / (4 * energy)
                    # if (i == 4) and (j == 0) and (k == 5):
                    #     print("Python Energy: ", energy, "LJ Grid: ", lj_grid[i][j][k])
    return lj_grid


def test_import():
    from Periodic import CrystalParams


def test_crystal_param():
    from Periodic import CrystalParams

    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 0.5, 0.5, 0],
            [0.5, 0, 0.5, 0],
            [0.5, 0.5, 0, 0],
            [0.5, 0.5, 0.5, 1],
        ]
    )
    elements = np.array([0, 0, 0, 0, 1])
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
    print(params.get_cartesian_coords())
    print(params.get_channels())
    print(params.get_channels().shape)

    print(params.get_elements())
    print(params.get_transform_matrix())

    ground_truth_energy_grid = generate_LJ_grid(
        params.get_cartesian_coords(),
        params.get_elements(),
        params.get_transform_matrix(),
        grid_size=32,
    )

    energy_grid = params.LJ_Grid(32)

    print(np.allclose(energy_grid, ground_truth_energy_grid, rtol=1e-03))
    np.save("ground_truth_energy_grid.npy", ground_truth_energy_grid)
    np.save("energy_grid.npy", energy_grid)

    # energy_grid = params.LJ_Grid(32);

    # print(type(energy_grid))
    # print(energy_grid.shape)
    # print(energy_grid[0][0])


if __name__ == "__main__":
    test_import()
    test_crystal_param()

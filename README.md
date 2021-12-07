# ParticleGrid

[![build Actions Status](https://github.com/ParticleGrid/ParticleGrid/workflows/build/badge.svg)](https://github.com/ParticleGrid/ParticleGrid/actions)

# Description

ParticleGrid is a SIMD accelerated 3D grid generation library. 3D grids are generated from conformed molecular points and can be used with conventional 3D deep learning tools. 
 

# Visualizations
![Molecule 1](/docs/images/real_mol_3.png)
![Molecule 2](/docs/images/real_mol_4.png)
# Installation 

## Pip

Install via pip with:

```
pip install git+https://github.com/ParticleGrid/ParticleGrid.git
```

## Source 

Clone and build repo with: 
```
git clone https://github.com/ParticleGrid/ParticleGrid.git
cd ParticleGrid
pip install .
```

# How to Use

Generating a grid from 3D coordinates: 

```python

import numpy as np
from ParticleGrid import coord_to_grid

# points are in the format (channel, x, y, z)
test_points = np.array([0, 0.5, 0.5, 0.5],
                       [1, 0.0, 0.1, 0.2])

grid = coord_to_grid(test_points,
                     width=1,
                     height=1,
                     depth=1,
                     num_channels=2,
                     grid_size=32,
                     variance=0.05)

print(grid.shape)  # Generates a (2,32,32,32) grid
```
More example uses of ParticleGrid can be found in the [examples](examples) directory.

# Upcoming Features

- [ ] Discretization 
- [ ] Per-atom weights and variance
- [ ] 2D grids
- [ ] Periodic crystal grids
- [ ] Multi-threading support 
- [ ] GPU support 
- [ ] PyTorch integration 

# License

ParticleGrid has a BSD-style license, as found in the [LICENSE](LICENSE) file.
 

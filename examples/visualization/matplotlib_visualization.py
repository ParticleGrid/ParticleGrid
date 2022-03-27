import os
import os.path as osp
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import GridGenerator as gg

data_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..')

def visualize(size, variance=0.05):
    data_file = osp.join(data_dir,'100_molecules.pickle')
    molecule = []
    print("Loading structure data ...")
    with open(data_file, 'rb') as f:
        data_loaded = pickle.load(f)
        molecule = data_loaded[0]
    print("Finished loading data. ")
    print(f"Number of atoms: {len(molecule)}")
    print("Creating visualization...")
    tensor = gg.molecule_grid(molecule, size, 8, variance*16)
    first_channel = tensor[0]
    fig3d = plt.figure(figsize=plt.figaspect(1))
    axes3d = fig3d.add_subplot( projection='3d')
    points = []
    for iz, iy, ix in np.ndindex(first_channel.shape):
        val = first_channel[ix, iy, iz];
        if val > 0.0:
            points.append(np.array([ix, iy, iz, val]))
    points = np.array(points)
    if len(points) > 0:
        cmap = give_colormap_transparency(plt.cm.hot)
        axes3d.scatter3D(points[:,0], points[:,1], points[:,2], c = points[:,3], cmap=cmap)
        margin = 2
        axes3d.set_xlim3d(left=margin, right=size-margin)
        axes3d.set_ylim3d(bottom=margin, top=size-margin)
        axes3d.set_zlim3d(bottom=margin, top=size-margin)
    plt.show()
    print("done")

def give_colormap_transparency(cmap):
    my_cmap = cmap(np.arange(cmap.N))
    transparent_size = len(my_cmap)//2
    my_cmap[:transparent_size,-1] = np.linspace(0.1, 1, transparent_size)
    my_cmap = ListedColormap(my_cmap)
    return my_cmap

if __name__ == "__main__":
    visualize(64)

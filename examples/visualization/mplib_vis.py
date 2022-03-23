

import pickle
import argparse

import os
import os.path as osp
import time

import numpy as np
import random as r

import multiprocessing as mp
import GridGenerator as gg
# from Discretizer import optimize_coords

_cur_dir = osp.dirname(osp.realpath(__file__))

_data_dir = osp.join(_cur_dir)

def benchmark(sizes, variance=0.1):

    _data_file = osp.join(_data_dir,'100_molecules.pickle')
    
    mol_data = []
    print("Loading structure data ...")
    with open(_data_file, 'rb') as f:
        _data_loaded = pickle.load(f)
        mol_data = _data_loaded[:1]
            
    print("Finished loading data. ")
    print("Number of molecules: {}".format(len(mol_data)))

    ctensors = []
    ctimes = []
    total = 0
    for mol in mol_data:
        total += len(mol)
    print(f"Total number of atoms: {total}")
    start_time = time.time()
    for size in sizes:
        print("Doing c-based generation for {}^3, {} molecules...".format(size, len(mol_data)))
        a = time.time_ns()
        ctensors.append(gg.list_grid(mol_data, size, 8, variance*16))
        # for m in mol_data:
            # ctensors.append(gg.molecule_grid(m, size, 8, variance*16))
        b = time.time_ns()
        diff = (b-a)/(10**9)
        print(diff, "s = ", len(mol_data)/diff, "molecules/s")
        ctimes.append(diff)
        all_points = []
        for tensor in ctensors[-1]:
            fig3d = plt.figure(figsize=plt.figaspect(1))
            for c in range(1):
                channel_tensor = tensor[c]
                points = []
                for iz, iy, ix in np.ndindex(channel_tensor.shape):
                    val = channel_tensor[ix, iy, iz];
                    if val > 0.0:
                        points.append(np.array([ix, iy, iz, val]))
                points = np.array(points)
                if len(points > 0):
                    all_points.append(points)
            for c in range(len(all_points)):
                points = all_points[c]
                axes3d = fig3d.add_subplot( projection='3d')
                if len(points) > 0:
                    cmap = plt.cm.hot
                    my_cmap = cmap(np.arange(cmap.N))
                    transparent_size = len(my_cmap)//4
                    my_cmap[:transparent_size,-1] = np.linspace(0.1, 1, transparent_size)
                    my_cmap = ListedColormap(my_cmap)
                    axes3d.scatter3D(points[:,0], points[:,1], points[:,2], c = points[:,3], cmap=my_cmap)
                    margin = 2
                    axes3d.set_xlim3d(left=margin, right=size-margin)
                    axes3d.set_ylim3d(bottom=margin, top=size-margin)
                    axes3d.set_zlim3d(bottom=margin, top=size-margin)
        plt.show()
        # gg.display_tensor(ctensors[-1][0], 1)
        # for mol in mol_data[:1]:
            # print("# atoms:", len(mol))
        print()
    # with open('grid_generation_times.log','w') as f:
        # f.write('Grid Size, C Version\n')
        # for grid_size, c_time in zip(sizes, ctimes):
            # f.write(f'{grid_size},{c_time}\n')	 
    end_time = time.time()
    print("generation done in", end_time - start_time, "seconds")
    

# benchmark([16, 17, 20, 23, 32, 48, 64, 128, 192])
# benchmark([16, 16, 20, 32, 48, 64, 128, 192])
# benchmark([16, 32, 48, 64, 128, 192])
# benchmark([16, 20, 32])
benchmark([32])
import os
import os.path as osp
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import GridGenerator as gg

data_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..')

def visualize(size, variance=0.25):
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
    fig3d = plt.figure(figsize=plt.figaspect(1))
    colormaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'GnBu', 'YlGn', 'RdPu']
    for i in range(len(colormaps)):
        plot = mlab.contour3d(tensor[i], transparent=True, colormap=colormaps[i])
        give_colormap_transparency(plot)
    mlab.outline()
    mlab.show()
    print("done")

def give_colormap_transparency(plot):
    lut = plot.module_manager.scalar_lut_manager.lut.table.to_array()
    lut[:, -1] = np.linspace(50, 255, 256)
    plot.module_manager.scalar_lut_manager.lut.table = lut

if __name__ == "__main__":
    visualize(64)

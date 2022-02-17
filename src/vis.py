

import pickle
import argparse

import os
import os.path as osp
import time

import numpy as np
import random as r
# import matplotlib.pyplot as plt
from mayavi import mlab

import multiprocessing as mp
import GridGenerator as gg
# from Discretizer import optimize_coords

_cur_dir = osp.dirname(osp.realpath(__file__))

_data_dir = osp.join(_cur_dir)

def benchmark(sizes, variance=0.06):

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
        colormaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'RdYlGn', 'RdBu', 'YlGn', 'RdPu']
        for i in range(8):
            print(colormaps[i])
            s = mlab.contour3d(ctensors[-1][0][i], transparent=True, colormap=colormaps[i])
            lut = s.module_manager.scalar_lut_manager.lut.table.to_array()
            lut[:, -1] = np.linspace(50, 255, 256)
            s.module_manager.scalar_lut_manager.lut.table = lut
        mlab.outline()
        mlab.show()
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
benchmark([128])

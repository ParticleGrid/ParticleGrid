

import pickle
import argparse

import os
import os.path as osp
import time

import multiprocessing as mp
import GridGenerator as gg

import numpy as np

import random as r

_cur_dir = osp.dirname(osp.realpath(__file__))

_data_dir = osp.join(_cur_dir)

def benchmark(sizes, variance=0.001):

    _data_file = osp.join(_data_dir,'100_molecules.pickle')
    
    mol_data = []
    print("Loading structure data ...")
    with open(_data_file, 'rb') as f:
        _data_loaded = pickle.load(f)
        mol_data = _data_loaded
            
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
        ctensors.append(gg.generate_grid(mol_data, size, size, size, 8, variance*16/size))
        b = time.time_ns()
        diff = (b-a)/(10**9)
        print(diff, "s = ", len(mol_data)/diff, "molecules/s")
        ctimes.append(diff)
        # gg.display_tensor(ctensors[-1], 8)
        # for mol in mol_data[:1]:
            # print("# atoms:", len(mol))
        print()
    # with open('grid_generation_times.log','w') as f:
        # f.write('Grid Size, C Version\n')
        # for grid_size, c_time in zip(sizes, ctimes):
            # f.write(f'{grid_size},{c_time}\n')	 
    end_time = time.time()
    print("generation done in", end_time - start_time, "seconds")
    

# benchmark([16, 16, 20, 32, 48, 64, 128, 192])
benchmark([16, 16, 32, 48, 64, 128, 192])
# benchmark([16, 20, 32, 48, 64, 128])
# benchmark([16, 20])


import os
import os.path as osp
import pickle
import numpy as np

from mayavi import mlab
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

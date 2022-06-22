from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from ParticleGrid import coord_to_grid
from ogb.lsc import PCQM4Mv2Dataset
import pickleslicer

class PCQM4M_Dataset(Dataset):
  def __init__(self,
               grid_size=32,
               variance=0.3,
               padding=2) -> None:
    super().__init__()
    self._targets = pickleslicer.load("targets.pickle")
    self._data = pickleslicer.load("mols.pickle")
    self.grid_size = grid_size
    self.variance = variance
    self.padding = padding

  def __len__(self):
    return 3378606

  def __getitem__(self, index):
    mol = self._data[index]
    mol_coords = mol[:, 1:] 
    depth, height, width = mol_coords.max(axis=0) + 2
    _data = coord_to_grid(mol,
                          width=width,
                          height=height,
                          depth=depth,
                          num_channels=7,
                          grid_size=self.grid_size,
                          variance=self.variance)
    y = self._targets[index]
    return torch.from_numpy(_data).float(), y

if __name__ == '__main__':
  dataset = PCQM4M_Dataset()
  from tqdm import tqdm
 
  for i in tqdm(dataset):
    grid, y = i

  
  
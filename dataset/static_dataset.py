from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from utils import helpers
from path import Path

class StaticModelNetDataset(data.Dataset):
    def __init__(self, root_dir, device, folder="train"):
        self.root_dir = Path(root_dir)
        self.pointclouds = torch.from_numpy(np.load(root_dir/Path(folder)/"pointclouds.npy").astype(np.float32))
        self.labels = torch.from_numpy(np.load(root_dir/Path(folder)/"labels.npy").astype(np.int32))
        self.pointclouds = self.pointclouds.to(device)
        self.labels = self.labels.to(device)

        assert(self.labels.size(0) == self.pointclouds.size(0))
        assert(self.pointclouds.size(2) == 3)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return self.pointclouds[idx,:,:], self.labels[idx].item()

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    helpers.gen_modelnet_id(datapath)
    d = StaticModelNetDataset(root=datapath)
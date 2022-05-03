from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from utils import helpers
from path import Path

class DynamicModelNetDataset(data.Dataset):
    def __init__(self, root_dir, folder="train", npoints=1024, transform=None):
        self.root_dir = Path(root_dir)
        folders = [dir for dir in sorted(os.listdir(self.root_dir)) if os.path.isdir(self.root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if transform else helpers.default_transforms(npoints)
        print(self.transforms)
        self.files = []
        for category in self.classes.keys():
            new_dir = self.root_dir/Path(category)/folder
            for file in os.listdir(new_dir/"verts"):
                if file.endswith('.npy'):
                    sample = {}
                    sample["filename"] = file
                    sample['pcd_path'] = new_dir
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]['filename']
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        pointcloud = helpers.extract_pointcloud(pcd_path, filename, self.transforms)
        return pointcloud, self.classes[category]

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    helpers.gen_modelnet_id(datapath)
    d = DynamicModelNetDataset(root=datapath)
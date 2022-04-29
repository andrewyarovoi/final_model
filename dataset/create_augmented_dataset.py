import numpy as np
import math
import os
import os.path
import sys
from path import Path
from utils import helpers

def count_number_of_files(root_dir, sub_f):
    count = 0

    # find all folders and assume that the folder name is the class names
    folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
    for category in folders:
        cat_dir = root_dir/Path(category)/Path(sub_f)/"verts"
        count += len(os.listdir(cat_dir))
    
    return count


def create_augmented_dataset(input_path, output_path, num_points=1024, num_augmentations = 1, sub_f="train"):
    # create path variables for input and output directories
    root_dir = Path(input_path)
    root_out_dir = Path(output_path)/sub_f
    if not os.path.exists(root_out_dir):
        os.makedirs(root_out_dir)
    
    # count the number of total files
    num_files = count_number_of_files(root_dir, sub_f)*num_augmentations

    # find all folders and assume that the folder name is the class names
    folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}

    # create numpy arrays to store labels and pointclouds
    labels = np.empty(num_files)
    pointclouds = np.empty((num_files, num_points, 3))
    idx = 0
    
    print("Sampling from ", sub_f, " folders in ", input_path)
    print("Found ", num_files, " to convert...")
    print("Progress:")
    num_updates = 1000.0
    update_idx = num_files/num_updates

    # run for multiple iterations to increase variety
    for i in range(num_augmentations):
        # iterate for each category/subf/verts/file
        for category in classes.keys():
            # create path variables for input
            cat_dir = root_dir/Path(category)/sub_f
            file_dir = cat_dir/"verts"

            # for each file
            for file in os.listdir(file_dir):
                labels[idx] = classes[category]
                pointclouds[idx, :, :] = helpers.extract_pointcloud(cat_dir, file, helpers.augment_transforms(num_points))
                
                idx += 1
                # used for terminal output to show progress
                if (idx > update_idx):
                    update_idx += num_files/num_updates
                    print(round((idx/num_files)*100.0, 2), "%", flush=True)

    print("done", flush=True)    

    print("Saving to: ", root_out_dir)
    np.save(root_out_dir/"pointclouds", pointclouds)
    np.save(root_out_dir/"labels", labels)

if __name__ == '__main__':
    create_augmented_dataset("../../data/ModelNet40_numpy/", "../../data/ModelNet40_aug/", num_augmentations = 1, sub_f="test")
    create_augmented_dataset("../../data/ModelNet40_numpy/", "../../data/ModelNet40_aug/", num_augmentations = 10, sub_f="train")
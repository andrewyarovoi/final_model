import numpy as np
import math
import os
import os.path
import sys
from path import Path

def convert_off_file(file):
    # checks if file has the correct header and parses the first line to get number of vertices and faces
    first_line = file.readline().strip()
    if 'OFF' == first_line:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    elif first_line[0:3] == "OFF":
        # we must check this condition because some of the files have a typo and are missing a '\n' after OFF
        n_verts, n_faces, __ = tuple([int(s) for s in first_line[3:].split(' ')])
    else:
        raise('Not a valid OFF header')
    
    # parse the rest of the file and store the values in numpy arrays
    verts = np.empty((n_verts,3), dtype=float)
    faces = np.empty((n_faces,3), dtype=int)
    for i in range(n_verts):
        j = 0
        for s in file.readline().strip().split(' '):
            verts[i, j] = float(s)
            j += 1

    for i in range(n_faces):
        j = 0
        for s in file.readline().strip().split(' ')[1:]:
            faces[i, j] = int(s)
            j += 1

    # return the vertices and faces as numpy arrays
    return verts, faces


def convert_dataset(input_path, output_path):
    # create path variables for input and output directories
    root_dir = Path(input_path)
    root_out_dir = Path(output_path)
    if not os.path.exists(root_out_dir):
        os.makedirs(root_out_dir)
    
    # find all folders and assume that the folder name is the class names
    folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}

    # iterate for each category
    for category in classes.keys():
        # create path variables for input and output directories for each class
        cat_dir = root_dir/category
        cat_out_dir = root_out_dir/category
        if not os.path.exists(cat_out_dir):
            os.makedirs(cat_out_dir)
        
        # iterate through the subfolders (train, test)
        subfolders = [dir for dir in sorted(os.listdir(cat_dir)) if os.path.isdir(cat_dir/dir)]
        for sub_f in subfolders:
            # create path variables for input directory (ex. root/table/train/) and output directories (one for verts and one for faces)
            new_dir = cat_dir/sub_f
            new_out_dir_verts = cat_out_dir/Path(sub_f)/Path("verts")
            new_out_dir_faces = cat_out_dir/Path(sub_f)/Path("faces")
            if not os.path.exists(new_out_dir_verts):
                os.makedirs(new_out_dir_verts)
            if not os.path.exists(new_out_dir_faces):
                os.makedirs(new_out_dir_faces)

            count = 0.0
            num_files = len(os.listdir(new_dir))
            print("converting ", category, "/", sub_f, ": ", end="", flush=True)
            # for each file
            for file in os.listdir(new_dir):
                count += 1.0
                # used for terminal output to show progress
                if (count > num_files/10.0):
                    count -= num_files/10.0
                    print(".", end="", flush=True)
                # actually converts the file and saves it in new format
                if file.endswith('.off'):
                    pcd_path = new_dir/file
                    with open(pcd_path, 'r') as f:
                        verts, faces = convert_off_file(f)
                        new_filename = file.split(".")[0]
                        np.save(new_out_dir_verts/new_filename, verts)
                        np.save(new_out_dir_faces/new_filename, faces)
            print("done", flush=True)


if __name__ == '__main__':
    convert_dataset("../../data/ModelNet40/", "../../data/ModelNet40_numpy/")
import numpy as np
import math
import random
import torch
from path import Path
import os
import os.path
from torchvision import transforms

def default_transforms(npoints):
    return transforms.Compose([
                                PointSampler(npoints),
                                Normalize(),
                                ToTensor()
                              ])

def augment_transforms(npoints):
    return transforms.Compose([
                                PointSampler(npoints),
                                RandRotation_z(),
                                Normalize(),
                                RandomNoise()
                              ])

def static_transforms(npoints):
    return transforms.Compose([
                                PointSampler(npoints),
                                Normalize()
                              ])

def extract_pointcloud(path, filename, transforms):
    verts = np.load(path/Path("verts")/filename)
    faces = np.load(path/Path("faces")/filename)
    pointcloud = transforms((verts, faces))
    return pointcloud

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))
        
class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        

    def __call__(self, mesh):
        verts, faces = mesh
        areas = np.zeros((faces.shape[0]))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i, 0], :],
                                           verts[faces[i, 1], :],
                                           verts[faces[i, 2], :]))
        
        sampled_faces = (np.random.choice(faces.shape[0],
                                      size=self.output_size,
                                      replace=True,
                                      p=areas/np.sum(areas)))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(self.output_size):
            sampled_points[i] = (self.sample_point(verts[faces[sampled_faces[i], 0], :],
                                                   verts[faces[sampled_faces[i], 1], :],
                                                   verts[faces[sampled_faces[i], 2], :]))
        
        return sampled_points

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud
        
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

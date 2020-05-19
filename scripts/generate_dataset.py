#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script that uses the cpp library SpherePlanning to sample paths
whose starts and goals are on a unit sphere.
"""

import h5py
import numpy as np
import open3d
from tqdm import tqdm

try:
    from SpherePlanning import SpherePlanning
except ImportError:
    SpherePlanning = None
    print("Cannot import SpherePlanning. Please check if you compile the src correctly.")


def sample_obstacle(num=20000):
    if SpherePlanning is None:
        raise ImportError
    sphere_planning = SpherePlanning()
    collision = np.zeros([num, 3])
    for i in tqdm(range(num)):
        while True:
            collision[i] = np.random.randn(3)
            collision[i] /= np.linalg.norm(collision[i])
            if not sphere_planning.isValid(collision[i]):
                break
    with h5py.File("data/paths.hdf5", 'a') as h5_file:
        h5_file.create_dataset("collision", data=collision)


def convert_obstacle_format():
    file_name = "./data/paths.hdf5"
    dataset_name = "collision"
    voxel_size = 0.05

    pcd = open3d.geometry.PointCloud()
    with h5py.File(file_name, 'r') as h5_file:
        pcd.points = open3d.utility.Vector3dVector(np.array(h5_file[dataset_name]))
    voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd,
                                                                                 voxel_size=voxel_size,
                                                                                 min_bound=np.array([-1, -1, -1]),
                                                                                 max_bound=np.array([1, 1, 1]))
    # convert to numpy
    np_grid = np.zeros(shape=(40, 40, 40))
    for voxel in voxel_grid.voxels:
        np_grid[tuple(voxel.grid_index)] = 1
    # save to hdf5
    with h5py.File("./data/dataset.hdf5", 'a') as h5_file:
        h5_file.create_dataset("voxels", data=np_grid)


def sample_path(num=20000):
    if SpherePlanning is None:
        raise ImportError
    sphere_planning = SpherePlanning()
    with h5py.File("data/paths.hdf5", 'a') as h5_file:
        all_datasets = list(h5_file.keys())
    for i in tqdm(range(num)):
        if "path%d" % i not in all_datasets:
            while True:
                start = np.random.randn(3)
                start /= np.linalg.norm(start)
                if sphere_planning.isValid(start):
                    break
            while True:
                goal = np.random.randn(3)
                goal /= np.linalg.norm(goal)
                if sphere_planning.isValid(goal):
                    break
            sphere_planning.plan(start, goal)
            path = sphere_planning.getPath()
            with h5py.File("data/paths.hdf5", 'a') as h5_file:
                h5_file.create_dataset("path%d" % i, data=path)
            sphere_planning.clear()


def convert_path_format():
    # allocate space
    total_length = 0
    with h5py.File("data/paths.hdf5", "r") as h5_file:
        for i in range(20000):
            total_length += h5_file["path%d" % i].len()-1
    all_input = np.zeros(shape=(total_length, 6))
    all_output = np.zeros(shape=(total_length, 3))
    # convert to input-output pair
    offset = 0
    with h5py.File("data/paths.hdf5", "r") as h5_file:
        for i in tqdm(range(20000)):
            path = h5_file["path%d" % i]
            N = path.shape[0]-1
            goal = path[-1]
            all_input[offset:offset+N, 0:3] = path[:N]
            all_input[offset:offset+N, 3:6] = goal
            all_output[offset:offset+N] = path[1:]
            offset += N
    # save to hdf5
    with h5py.File("./data/dataset.hdf5", 'a') as h5_file:
        h5_file.create_dataset("input", data=all_input)
        h5_file.create_dataset("output", data=all_output)


if __name__ == "__main__":
    convert_obstacle_format()

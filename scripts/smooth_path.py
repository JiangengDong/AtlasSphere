import SpherePlanning
import numpy as np
import h5py
from tqdm import tqdm
import open3d

def smoothPath(input_filename, output_filename):
    sp = SpherePlanning.SpherePlanning()
    for i in tqdm(range(20000)):
        with h5py.File(input_filename, 'r') as h5_file:
            path = np.array(h5_file["path%d" % i])
        new_path = sp.smoothPath(path)
        with h5py.File(output_filename, 'a') as h5_file:
            h5_file.create_dataset("path%d" % i, data=new_path)

def convert_path_format(input_filename, output_filename):
    # allocate space
    total_length = 0
    with h5py.File(input_filename, "r") as h5_file:
        for i in range(20000):
            total_length += h5_file["path%d" % i].len()-1
    all_input = np.zeros(shape=(total_length, 6))
    all_output = np.zeros(shape=(total_length, 3))
    # convert to input-output pair
    offset = 0
    with h5py.File(input_filename, "r") as h5_file:
        for i in tqdm(range(20000)):
            path = h5_file["path%d" % i]
            N = path.shape[0]-1
            goal = path[-1]
            all_input[offset:offset+N, 0:3] = path[:N]
            all_input[offset:offset+N, 3:6] = goal
            all_output[offset:offset+N] = path[1:]
            offset += N
    # save to hdf5
    with h5py.File(output_filename, 'a') as h5_file:
        h5_file.create_dataset("input", data=all_input)
        h5_file.create_dataset("output", data=all_output)

def convert_obstacle_format():
    file_name = "./data/training_samples/paths.hdf5"
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
    with h5py.File("./data/training_samples/new_dataset.hdf5", 'a') as h5_file:
        h5_file.create_dataset("voxels", data=np_grid)

if __name__ == "__main__":
    # smoothPath("data/training_samples/paths.hdf5", "data/training_samples/new_paths.hdf5")
    # convert_path_format("data/training_samples/new_paths.hdf5", "data/training_samples/new_dataset.hdf5")
    convert_obstacle_format()

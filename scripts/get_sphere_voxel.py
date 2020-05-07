import open3d
import numpy as np
import h5py

def load_voxel(file_name, dataset_name, voxel_size):
    pcd = open3d.geometry.PointCloud()
    with h5py.File(file_name, 'r') as h5_file:
        pcd.points = open3d.utility.Vector3dVector(np.array(h5_file[dataset_name]))
    return open3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

def main():
    voxel = load_voxel("./data/paths.hdf5", "collision", 0.05)
    open3d.visualization.draw_geometries([voxel])

if __name__ == "__main__":
    main()

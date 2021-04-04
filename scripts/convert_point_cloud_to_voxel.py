import open3d as o3d
import numpy as np
from tqdm import tqdm


def convert_point_cloud_to_voxel(point_cloud_path: str, voxel_path: str):
    points = np.load(point_cloud_path)
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pt, 0.05, np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    voxel_indices = np.stack([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    converted_voxel = np.zeros((40, 40, 40), dtype=np.float32)
    converted_voxel[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0
    np.save(voxel_path, converted_voxel)

def main():
    convert_point_cloud_to_voxel("./data/point_cloud/envOld.npy", "./data/voxel/envOld.npy")
    for i in tqdm(range(40)):
        convert_point_cloud_to_voxel("./data/point_cloud/env{}.npy".format(i), "./data/voxel/env{}.npy".format(i))

if __name__ == "__main__":
    main()

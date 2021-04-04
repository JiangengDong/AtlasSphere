# This script is used to visual the data for manual checking.
import open3d as o3d
import numpy as np


def createBrick(phi, theta, size=0.1, color=[0.0, 0.3, 0]):
    mesh = o3d.geometry.TriangleMesh.create_box(size, size, size)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.paint_uniform_color(color)

    mesh.translate(np.array([1-size / 2, -size / 2, -size / 2]))
    rot = np.array([[np.cos(theta) * np.sin(phi), -np.sin(theta), -np.cos(theta)*np.cos(phi)],
                    [np.sin(theta)*np.sin(phi), np.cos(theta), -np.sin(theta)*np.cos(phi)],
                    [np.cos(phi), 0, np.sin(phi)]])

    mesh.rotate(rot, center=(0, 0, 0))
    return mesh


def visualize_brick_config(brick_config_path):
    bricks = [createBrick(phi, theta) for (phi, theta) in np.load(brick_config_path)]

    sphere = o3d.geometry.TriangleMesh.create_sphere(1.0, 100)
    sphere.compute_vertex_normals()
    sphere.compute_triangle_normals()

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0)
    coord.compute_vertex_normals()
    coord.compute_triangle_normals()

    o3d.visualization.draw_geometries([sphere, coord] + bricks)


def visualize_start_and_goal(brick_config_path, start_path, goal_path):
    starts = o3d.geometry.PointCloud()
    starts.points = o3d.utility.Vector3dVector(np.load(start_path))
    starts.paint_uniform_color([0.7, 0.0, 0.0])

    goals = o3d.geometry.PointCloud()
    goals.points = o3d.utility.Vector3dVector(np.load(goal_path))
    goals.paint_uniform_color([0.0, 0.0, 0.7])

    bricks = [createBrick(phi, theta) for (phi, theta) in np.load(brick_config_path)]

    sphere = o3d.geometry.TriangleMesh.create_sphere(1.0, 100)
    sphere.compute_vertex_normals()
    sphere.compute_triangle_normals()

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0)
    coord.compute_vertex_normals()
    coord.compute_triangle_normals()

    o3d.visualization.draw_geometries([starts, goals, sphere, coord] + bricks)


def visualize_path(brick_config_path, path_path):
    path = o3d.geometry.LineSet()
    with np.load(path_path) as data:
        points = data["path_2"]
    N = points.shape[0]
    lines = np.stack([np.arange(N-1), np.arange(1, N)], axis=1).astype(np.int32)
    path.points = o3d.utility.Vector3dVector(points)
    path.lines = o3d.utility.Vector2iVector(lines)

    bricks = [createBrick(phi, theta) for (phi, theta) in np.load(brick_config_path)]

    sphere = o3d.geometry.TriangleMesh.create_sphere(1.0, 100)
    sphere.compute_vertex_normals()
    sphere.compute_triangle_normals()

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0)
    coord.compute_vertex_normals()
    coord.compute_triangle_normals()

    o3d.visualization.draw_geometries([path, sphere, coord] + bricks)


def visualize_point_cloud(brick_config_path, point_cloud_path):
    points = np.load(point_cloud_path)
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pt, 0.05, np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))

    bricks = [createBrick(phi, theta) for (phi, theta) in np.load(brick_config_path)]

    sphere = o3d.geometry.TriangleMesh.create_sphere(1.0, 100)
    sphere.compute_vertex_normals()
    sphere.compute_triangle_normals()

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0)
    coord.compute_vertex_normals()
    coord.compute_triangle_normals()

    o3d.visualization.draw_geometries([voxel_grid, sphere, coord] + bricks)


if __name__ == "__main__":
    # visualize_brick_config("./data/brick_config/env21.npy")
    # visualize_start_and_goal("./data/brick_config/env20.npy", "./data/train/env20_start.npy", "./data/train/env20_goal.npy")
    # visualize_path("./data/brick_config/env20.npy", "./data/train/env20_path.npz")
    visualize_point_cloud("./data/brick_config/env20.npy", "./data/point_cloud/env20.npy")

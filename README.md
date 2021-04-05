# AtlasSphere

This project aims to sample a batch of training paths for MPNet. It consists of two parts: C++ part and Python part.
The two part are connected by Pybind11. The C++ part exposes the interface for planning, which is then used by Python.

The C++ part uses OMPL to do path planning. The constraint and state validity checker are both defined in C++ part.

The Python part is used to store the sampled paths to a HDF5 file. Each path is a separate dataset with name "path\[num\]". There is also a dataset called "collision" that contains the collided configs.

## Dependencies

- Python: 3.5/3.7
- C++: 11
- OMPL: 1.4.2
- Boost: 1.58
- HDF5: 1.8.16
- YAML-CPP

## Steps for evaluating the MPNet algorithm

1. Train MPNet:
  1. Generate several environments with `GenerateBrickConfig.cpp`.
  1. Generate training samples:
    1. Generate 10000 pairs of starts and goals in these environments with `GenerateTrainingStartAndGoal.cpp`.
    1. Generate paths for these starts and goals with `GenerateTrainingPath.cpp`.
    1. Clean up training samples with `preprocess_path.py`.
  1. Generate voxels:
    1. Generate point clouds for these environments with `GeneratePointCloud.cpp`.
    1. Convert point clouds to voxels with `convert_point_cloud_to_voxel.py`.
  1. Train with `main.py`.
1. Test MPNet:
  1. Generate embedded voxel with `convert_voxel_to_embedded_voxel.py`.
  1. Generate 1000 pairs of testing starts and goals with `GenerateTestingStartAndGoal.cpp`.
  1. Evaluate RRTConnect's performance with `TestRRTConnect.cpp`.
  1. Evaluate CoMPNet's performance with `TestMPNet.cpp`.

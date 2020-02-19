# AtlasSphere

This project aims to sample a batch of training paths for MPNet. It consists of two parts: C++ part and Python part. 
The two part are connected by Pybind11. The C++ part exposes the interface for planning, which is then used by Python.

The C++ part uses OMPL to do path planning. The constraint and state validity checker are both defined in C++ part. 

The Python part is used to store the sampled paths to a HDF5 file. Each path is a separate dataset with name "path\[num\]". There is also a dataset called "collision" that contains the collided configs.

## Version of libraries

- Python: 3.5/3.7
- C++: 11
- OMPL: 1.4.2
- Boost: 1.58

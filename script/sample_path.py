#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script that uses the cpp library SpherePlanning to sample paths
whose starts and goals are on a unit sphere.
"""

import h5py
import numpy as np
from SpherePlanning import SpherePlanning


def main():
    """main function to sample paths."""
    num_path = 20
    sphere_planning = SpherePlanning()
    collisions = []
    with h5py.File("data/paths2.hdf5", 'w') as h5_file:
        for i in range(num_path):
            print("Sampling the %dth start and goal..." % i)
            while True:
                start = np.random.randn(3)
                start /= np.linalg.norm(start)
                if sphere_planning.isValid(start):
                    break
                collisions.append(start)
            while True:
                goal = np.random.randn(3)
                goal /= np.linalg.norm(goal)
                if sphere_planning.isValid(goal):
                    break
                collisions.append(goal)
            print("Planning for the %dth start and goal..." % i)
            if sphere_planning.plan(start, goal):
                path = sphere_planning.getPath()
                h5_file.create_dataset("path%d" % i, data=path)
            sphere_planning.clear()
        collisions = np.array(collisions)
        h5_file.create_dataset("collision", data=collisions)


if __name__ == "__main__":
    main()

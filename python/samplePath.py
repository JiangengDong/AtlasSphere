#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from SpherePlanning import SpherePlanning
import numpy as np
import h5py

def main():
    numPath = 20
    sp = SpherePlanning()
    collisions = []
    with h5py.File("data/paths.hdf5", 'w') as f:
        for i in range(numPath):
            print("Sampling the %dth start and goal..." % i)
            while True:
                start = np.random.randn(3)
                start /= np.linalg.norm(start)
                if sp.isValid(start):
                    break
                else:
                    collisions.append(start)
            while True:
                goal = np.random.randn(3)
                goal /= np.linalg.norm(goal)
                if sp.isValid(goal):
                    break
                else:
                    collisions.append(goal)
            print("Planning for the %dth start and goal..." % i)
            if sp.plan(start, goal):
                path = sp.getPath()
                f.create_dataset("path%d" % i, data=path)
        collisions = np.array(collisions)
        f.create_dataset("collision", data=collisions)

                

if __name__ == "__main__":
    main()

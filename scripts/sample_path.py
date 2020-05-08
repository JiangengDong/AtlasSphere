#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script that uses the cpp library SpherePlanning to sample paths
whose starts and goals are on a unit sphere.
"""

import h5py
import numpy as np
from SpherePlanning import SpherePlanning
from tqdm import tqdm


def sample():
    sphere_planning = SpherePlanning()
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
    return path


def main(n_total, index):
    partition = 20000//n_total
    with h5py.File("data/paths%d.hdf5" % index, 'a') as h5_file:
        for i in tqdm(range(partition*index, partition*(index+1))):
            path = sample()
            h5_file.create_dataset("path%d" % i, data=path)
            h5_file.flush()

if __name__ == "__main__":
    n_total = int(input("total number of process: "))
    index = int(input("index of current process: "))
    main(n_total, index)

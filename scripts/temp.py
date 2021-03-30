import numpy as np

for i in range(1):
    filename = "./data/train/env%d_path.npz" % i
    data = np.load(filename)["path_0"]
    print(data[:5])

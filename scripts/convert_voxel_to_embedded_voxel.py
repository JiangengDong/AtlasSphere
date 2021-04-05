import torch
import numpy as np
from tqdm import trange

enet = torch.jit.load("./data/pytorch_model/enet_script_gpu.pt")

voxels = []
for i in trange(50, desc="loading voxels"):
    voxels.append(np.load("./data/voxel/env{}.npy".format(i))[np.newaxis, ...].astype(np.float32))
voxels.append(np.load("./data/voxel/envOld.npy")[np.newaxis, ...].astype(np.float32))
voxels = torch.from_numpy(np.concatenate(voxels)).cuda()

with torch.no_grad():
    embedded_voxels = enet(voxels).cpu().numpy()
for i in trange(50, desc="saving embedded voxels"):
    np.save("./data/voxel/env{}_embedded.npy".format(i), embedded_voxels[i])
    np.savetxt("./data/voxel/env{}_embedded.csv".format(i), embedded_voxels[i])
np.save("./data/voxel/envOld_embedded.npy", embedded_voxels[-1])
np.savetxt("./data/voxel/envOld_embedded.csv", embedded_voxels[-1])

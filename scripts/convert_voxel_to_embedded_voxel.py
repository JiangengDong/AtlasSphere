import torch
import numpy as np

enet = torch.jit.load("./data/pytorch_model/enet_script_gpu.pt")

voxels = []
for i in range(10):
    voxels.append(np.load("./data/voxel/env{}.npy".format(i))[np.newaxis, ...].astype(np.float32))
voxels.append(np.load("./data/voxel/envOld.npy")[np.newaxis, ...].astype(np.float32))
voxels = torch.from_numpy(np.concatenate(voxels))

with torch.no_grad():
    embedded_voxels = enet.forward(voxels).cpu().numpy()
print(embedded_voxels.shape)
for i in range(10):
    np.save("./data/voxel/env{}_embedded.npy".format(i), embedded_voxels[i])
    np.savetxt("./data/voxel/env{}_embedded.csv".format(i), embedded_voxels[i])
np.save("./data/voxel/envOld_embedded.npy", embedded_voxels[-1])
np.savetxt("./data/voxel/envOld_embedded.csv", embedded_voxels[-1])

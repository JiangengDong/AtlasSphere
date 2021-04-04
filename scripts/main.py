import numpy as np
from models import MPNet
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


def get_data() -> [torch.Tensor]:
    inputs = []
    outputs = []
    voxel_idxs = []
    voxels = []
    for i in range(40):
        with np.load("./data/train/env{}_cleaned.npz".format(i)) as data:
            inputs.append(data["input"].astype(np.float32))
            outputs.append(data["output"].astype(np.float32))
            N = data["input"].shape[0]
            voxel_idxs.append(np.ones((N, ), dtype=np.int64) * i)
        voxels.append(np.load("./data/voxel/env{}.npy".format(i))[np.newaxis, ...].astype(np.float32))
    return [torch.from_numpy(np.concatenate(data)) for data in [inputs, outputs, voxel_idxs, voxels]]


def main():
    inputs, outputs, voxel_idxs, voxels = get_data()
    dataset = DataLoader(TensorDataset(inputs, outputs, voxel_idxs), batch_size=256, shuffle=True)
    # TODO: use ScriptModule from the beginning, so that we do not need to convert it later.
    mpnet = MPNet().cuda()
    optimizer = torch.optim.Adam(mpnet.parameters())
    writer = SummaryWriter("./data/tensorboard")

    def id_to_voxel(idxs: torch.Tensor) -> torch.Tensor:
        return torch.index_select(voxels, 0, idxs)

    idx = 0
    for x, y, z in tqdm(dataset):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        z = id_to_voxel(z).cuda()
        loss = F.mse_loss(y, mpnet.forward(x, z))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("loss", loss, idx)

        if idx % 200 == 0:
            mpnet.encoder.save("./data/pytorch_model/enet_script_gpu.pt")
            mpnet.pnet.save("./data/pytorch_model/pnet_script_gpu.pt")
            mpnet.save("./data/pytorch_model/mpnet_script_gpu.pt")

        idx += 1

    mpnet.encoder.save("./data/pytorch_model/enet_script_gpu.pt")
    mpnet.pnet.save("./data/pytorch_model/pnet_script_gpu.pt")
    mpnet.save("./data/pytorch_model/mpnet_script_gpu.pt")


if __name__ == "__main__":
    main()

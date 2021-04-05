import numpy as np
from models import MPNet
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


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
    now = datetime.now()
    unique_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join("./data/tensorboard", unique_name)
    save_dir = os.path.join("./data/pytorch_model", unique_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    inputs, outputs, voxel_idxs, voxels = get_data()
    dataset = DataLoader(TensorDataset(inputs, outputs, voxel_idxs), batch_size=256, shuffle=True)
    mpnet = MPNet().cuda()
    optimizer = torch.optim.Adam(mpnet.parameters())
    writer = SummaryWriter(log_dir)

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

        if idx % 1000 == 0:
            torch.save(mpnet.state_dict(), os.path.join(save_dir, "mpnet_weight_gpu.pt"))
            torch.jit.script(mpnet).save(os.path.join(save_dir, "mpnet_script_gpu.pt"))
            torch.jit.script(mpnet.encoder).save(os.path.join(save_dir, "enet_script_gpu.pt"))
            torch.jit.script(mpnet.pnet).save(os.path.join(save_dir, "pnet_script_gpu.pt"))

        idx += 1

    torch.save(mpnet.state_dict(), os.path.join(save_dir, "mpnet_weight_gpu.pt"))
    torch.jit.script(mpnet).save(os.path.join(save_dir, "mpnet_script_gpu.pt"))
    torch.jit.script(mpnet.encoder).save(os.path.join(save_dir, "enet_script_gpu.pt"))
    torch.jit.script(mpnet.pnet).save(os.path.join(save_dir, "pnet_script_gpu.pt"))


if __name__ == "__main__":
    main()

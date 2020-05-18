import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm

from models import VoxelEncoder, PNet


def load_dataset():
    with h5py.File("./data/dataset.hdf5", 'r') as h5_file:
        samples = torch.from_numpy(h5_file["input"].value.astype(np.float32))
        labels = torch.from_numpy(h5_file["output"].value.astype(np.float32))
        voxels = torch.from_numpy(np.expand_dims(h5_file["voxels"].value.astype(np.float32), 0))
    return samples, labels, voxels


def train(args):
    os.makedirs(args.model_path, exist_ok=True)

    # load data
    samples, labels, voxels = load_dataset()
    voxels = voxels.repeat_interleave(args.batch_size, dim=0)

    # build networks
    enet = VoxelEncoder(args.insz_enet, args.outsz_enet)
    pnet = PNet(args.insz_pnet, args.outsz_pnet)

    # move to cuda
    if torch.cuda.is_available():
        enet = enet.cuda()
        pnet = pnet.cuda()
        samples = samples.cuda()
        labels = labels.cuda()
        voxels = voxels.cuda()
        print("Using CUDA.")
    else:
        print("Using CPU.")

    # define loss
    criterion = torch.nn.MSELoss()
    params = list(enet.parameters()) + list(pnet.parameters())
    optimizer = torch.optim.Adagrad(params)

    # define dataset
    dataset = TensorDataset(samples, labels)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # initiate tensorboard
    writer = SummaryWriter("logs/sphere")

    for epoch in tqdm(range(args.num_epochs)):
        total_loss = 0
        for x, y in tqdm(data_loader):
            enet.zero_grad()
            pnet.zero_grad()

            embedded_voxels = enet.forward(voxels)
            input_pnet = torch.cat([embedded_voxels, x], 1)
            output_pnet = pnet.forward(input_pnet)

            loss = criterion(output_pnet, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        writer.add_scalar("loss", total_loss)
        if epoch % args.save_step == 0:
            enet_path = os.path.join(args.model_path, "enet%d.pkl" % epoch)
            torch.save(enet.state_dict(), enet_path)
            pnet_path = os.path.join(args.model_path, "pnet%d.pkl" % epoch)
            torch.save(pnet.state_dict(), pnet_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--insz_enet', type=int, default=40, help='dimension of ENets input vector')
    parser.add_argument('--outsz_enet', type=int, default=128, help='dimension of ENets output vector')

    parser.add_argument('--insz_pnet', type=int, default=134, help='dimension of PNets input vector')
    parser.add_argument('--outsz_pnet', type=int, default=3, help='dimension of PNets output vector')

    parser.add_argument('--num_epochs', type=int, default=401)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    train(args)

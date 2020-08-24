import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm

from models import VoxelEncoder, PNet, PNet_Annotated


def load_dataset():
    with h5py.File("./data/training_samples/new_dataset.hdf5", 'r') as h5_file:
        samples = torch.from_numpy(h5_file["input"][()].astype(np.float32))
        labels = torch.from_numpy(h5_file["output"][()].astype(np.float32))
        voxels = torch.from_numpy(np.expand_dims(h5_file["voxels"][()].astype(np.float32), 0))
        voxels = voxels.reshape([1, -1])
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
        for x, y in data_loader:
            enet.zero_grad()
            pnet.zero_grad()

            encoded_voxels = enet.forward(voxels)
            input_pnet = torch.cat([encoded_voxels, x], 1)
            output_pnet = pnet.forward(input_pnet)

            loss = criterion(output_pnet, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        writer.add_scalar("loss", total_loss, epoch)
        if epoch % args.save_step == 0:
            enet_path = os.path.join(args.model_path, "enet%d.pkl" % epoch)
            torch.save(enet.state_dict(), enet_path)
            pnet_path = os.path.join(args.model_path, "pnet%d.pkl" % epoch)
            torch.save(pnet.state_dict(), pnet_path)

    # save to torch script
    enet_scripted = torch.jit.script(enet)
    enet_scripted.save(os.path.join(args.model_path, "enet_script.pt"))
    pnet_scripted = torch.jit.script(pnet)
    pnet_scripted.save(os.path.join(args.model_path, "pnet_script.pt"))


def copypnet(MLP_to_copy, mlp_weights):
	# this function is where weights are manually copied from the originally trained
	# MPNet models (which have different naming convention for the weights that doesn't
	# work with manual dropout implementation) into the models defined in this script
	# which have the new layer naming convention

	# mlp_weights is just a state_dict() with the good model weights, not loaded into a particular model yet
	# MLP_to_copy is one of the MLP_Python models defined above (depending on 1.0 or 2.0)


	MLP_to_copy.state_dict()['fc1.0.weight'].copy_(mlp_weights['fc.0.weight'])
	MLP_to_copy.state_dict()['fc2.0.weight'].copy_(mlp_weights['fc.3.weight'])
	MLP_to_copy.state_dict()['fc3.0.weight'].copy_(mlp_weights['fc.6.weight'])
	MLP_to_copy.state_dict()['fc4.0.weight'].copy_(mlp_weights['fc.9.weight'])
	MLP_to_copy.state_dict()['fc5.0.weight'].copy_(mlp_weights['fc.12.weight'])
	MLP_to_copy.state_dict()['fc6.0.weight'].copy_(mlp_weights['fc.14.weight'])


	MLP_to_copy.state_dict()['fc1.0.bias'].copy_(mlp_weights['fc.0.bias'])
	MLP_to_copy.state_dict()['fc2.0.bias'].copy_(mlp_weights['fc.3.bias'])
	MLP_to_copy.state_dict()['fc3.0.bias'].copy_(mlp_weights['fc.6.bias'])
	MLP_to_copy.state_dict()['fc4.0.bias'].copy_(mlp_weights['fc.9.bias'])
	MLP_to_copy.state_dict()['fc5.0.bias'].copy_(mlp_weights['fc.12.bias'])
	MLP_to_copy.state_dict()['fc6.0.bias'].copy_(mlp_weights['fc.14.bias'])


	MLP_to_copy.state_dict()['fc1.1.weight'].copy_(mlp_weights['fc.1.weight'])
	MLP_to_copy.state_dict()['fc2.1.weight'].copy_(mlp_weights['fc.4.weight'])
	MLP_to_copy.state_dict()['fc3.1.weight'].copy_(mlp_weights['fc.7.weight'])
	MLP_to_copy.state_dict()['fc4.1.weight'].copy_(mlp_weights['fc.10.weight'])
	MLP_to_copy.state_dict()['fc5.1.weight'].copy_(mlp_weights['fc.13.weight'])


	return MLP_to_copy


def copy_pnet_weight(destination, source):
    # copy linear layers' weights
    destination["fc1.0.weight"].copy_(source["fc.0.weight"])
    destination["fc2.0.weight"].copy_(source["fc.3.weight"])
    destination["fc3.0.weight"].copy_(source["fc.6.weight"])
    destination["fc4.0.weight"].copy_(source["fc.9.weight"])
    destination["fc5.0.weight"].copy_(source["fc.12.weight"])
    destination["fc6.0.weight"].copy_(source["fc.14.weight"])

    # copy linear layers' biases
    destination["fc1.0.bias"].copy_(source["fc.0.bias"])
    destination["fc2.0.bias"].copy_(source["fc.3.bias"])
    destination["fc3.0.bias"].copy_(source["fc.6.bias"])
    destination["fc4.0.bias"].copy_(source["fc.9.bias"])
    destination["fc5.0.bias"].copy_(source["fc.12.bias"])
    destination["fc6.0.bias"].copy_(source["fc.14.bias"])

    # copy PReLU layers' weights
    destination["fc1.1.weight"].copy_(source["fc.1.weight"])
    destination["fc2.1.weight"].copy_(source["fc.4.weight"])
    destination["fc3.1.weight"].copy_(source["fc.7.weight"])
    destination["fc4.1.weight"].copy_(source["fc.10.weight"])
    destination["fc5.1.weight"].copy_(source["fc.13.weight"])


def export():
    pnet = PNet_Annotated()
    copy_pnet_weight(pnet.state_dict(), torch.load("./models/pnet400.pkl"))
    pnet.cuda()

    inp = torch.rand(1, 134).cuda()
    pnet(inp)
    pnet.save("./models/pnet.pt")

    pnet = torch.jit.load("./models/pnet.pt")
    pnet.cuda()
    print(pnet)


def encode_voxel():
    enet = VoxelEncoder(40, 128)
    enet.load_state_dict(torch.load("./models/enet400.pkl"))
    _, _, voxels = load_dataset()
    encoded_voxels = enet.forward(voxels)
    np.savetxt("./models/encoded_voxels.csv", encoded_voxels.detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('task', type=str, choices=("train", "test"), help='task to perform')
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--insz_enet', type=int, default=64000, help='dimension of ENets input vector')
    parser.add_argument('--outsz_enet', type=int, default=128, help='dimension of ENets output vector')

    parser.add_argument('--insz_pnet', type=int, default=134, help='dimension of PNets input vector')
    parser.add_argument('--outsz_pnet', type=int, default=3, help='dimension of PNets output vector')

    parser.add_argument('--num_epochs', type=int, default=401)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    export()

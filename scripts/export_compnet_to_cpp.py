from __future__ import division

import torch
import torch.nn as nn
import pickle
from torch.autograd import Variable
import numpy as np
import copy
from data_loader import load_test_dataset
from models import VoxelEncoder, PNet,Enet_constraint

'''enet_orig = VoxelEncoder(33, 256)
pnet_orig = PNet(302, 7)
enet_c_orig=Enet_constraint(5,32)

pnet_orig.load_state_dict(torch.load('models/6_B2_4Best/pnet400.pkl'))
enet_orig.load_state_dict(torch.load('models/6_B2_4Best/enet400.pkl'))
enet_c_orig.load_state_dict(torch.load('models/6_B2_4Best/enet_c400.pkl'))

enet_orig.cuda()
pnet_orig.cuda()
enet_c_orig.cuda()'''


'''class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=33, out_channels=64, kernel_size=[5,5], stride=[2,2]),
			nn.PReLU(),
			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3,3], stride=[1,1]),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size=[2,2])
		)
		self.head = nn.Sequential(
			nn.Linear(1152, 256),
			nn.PReLU(),
			nn.Linear(256, output_size))
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x'''


class VoxelEncoder_Annotated(torch.jit.ScriptModule):
	__constants__ = ['encoder']

	def __init__(self):
		super(VoxelEncoder_Annotated, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=33, out_channels=64, kernel_size=[5,5], stride=[2,2]),
			nn.PReLU(),
			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3,3], stride=[1,1]),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size=[2,2])
		)
		self.head = nn.Sequential(
			nn.Linear(1152, 256),
			nn.PReLU(),
			nn.Linear(256, 256))

	@torch.jit.script_method
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x

class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=33, out_channels=64, kernel_size=[5,5], stride=[2,2]),
			nn.PReLU(),
			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3,3], stride=[1,1]),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size=[2,2])
		)
		self.head = nn.Sequential(
			nn.Linear(1152, 256),
			nn.PReLU(),
			nn.Linear(256, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x




class OHEncoder(nn.Module):
    def __init__(self):
        super(OHEncoder, self).__init__()
        self.encoder_c = nn.Sequential(nn.Linear(5, 128), nn.PReLU(),nn.Linear(128, 32))

    def forward(self, x):
        x = self.encoder_c(x)
        return x

# This is used for creating the encoder model that is exported with the MPNet1.0 model
class OneHotEncoder_Annotated(torch.jit.ScriptModule):
	__constants__ = ['encoder_c']

	def __init__(self):
		super(OneHotEncoder_Annotated, self).__init__()
		self.fc = nn.Sequential(nn.Linear(5, 128), nn.PReLU(),nn.Linear(128, 32))

	@torch.jit.script_method
	def forward(self, x):
		x = self.fc(x)
		return x

# This is the MLP model that is trained during normal MPNet1.0 training, without the dropout layers
# and with different names for each layer. It is used to manually copy the weights from the model that was
# originally trained such that the layer names match when copying from THIS model to the ANNOTATED Model


'''class Pnet(nn.Module):
	def __init__(self):
		super(Pnet, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(398, 1280), nn.PReLU())
		self.fc2 = nn.Sequential(nn.Linear(1280, 896), nn.PReLU())
		self.fc3 = nn.Sequential(nn.Linear(896, 512), nn.PReLU())
		self.fc4 = nn.Sequential(nn.Linear(512, 384), nn.PReLU())
		self.fc5 = nn.Sequential(nn.Linear(384, 256), nn.PReLU())
		self.fc6 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
		self.fc7 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
		self.fc8 = nn.Sequential(nn.Linear(64, 32), nn.PReLU())
		self.fc9 = nn.Linear(32, 7)

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		out = self.fc5(out)
		out = self.fc6(out)
		out = self.fc7(out)
		out = self.fc8(out)
		out = self.fc9(out)
		return out'''



'''class Pnet(nn.Module):
	def __init__(self):
		super(Pnet, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(398, 896), nn.PReLU())
		self.fc2 = nn.Sequential(nn.Linear(896, 512), nn.PReLU())
		self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
		self.fc4 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
		self.fc5 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
		self.fc6 = nn.Sequential(nn.Linear(64, 32), nn.PReLU())
		self.fc7 = nn.Sequential(nn.Linear(32, 7))

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		out = self.fc5(out)
		out = self.fc6(out)
		out = self.fc7(out)
		return out'''


'''class Pnet(nn.Module):
	def __init__(self):
		super(Pnet, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(398, 512), nn.PReLU())
		self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
		self.fc3 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
		self.fc4 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
		self.fc5 = nn.Sequential(nn.Linear(64, 32), nn.PReLU())
		self.fc6 = nn.Sequential(nn.Linear(32, 7))

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		out = self.fc5(out)
		out = self.fc6(out)
		return out'''

class Pnet(nn.Module):
	def __init__(self):
		super(Pnet, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(398, 896), nn.PReLU())
		self.fc2 = nn.Sequential(nn.Linear(896, 512), nn.PReLU())
		self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
		self.fc4 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
		self.fc5 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
		self.fc6 = nn.Sequential(nn.Linear(64, 7))

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		out = self.fc5(out)
		out = self.fc6(out)
		return out



# This is the MLP model that is annotated using TorchScript, with dropout layers manually implemented
# in the forward pass. The parameters for each layer are copied in from the MLP_Python model (above)
#__constants__ = ['fc1','fc2','fc3','fc4','fc5','fc6','fc7','device']
class Pnet_Annotated(torch.jit.ScriptModule):
	__constants__ = ['fc1','fc2','fc3','fc4','fc5','fc6','device']
	def __init__(self):
		super(Pnet_Annotated, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(398, 896), nn.PReLU())
		self.fc2 = nn.Sequential(nn.Linear(896, 512), nn.PReLU())
		self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
		self.fc4 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
		self.fc5 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
		self.fc6 = nn.Sequential(nn.Linear(64, 7))

		self.device = torch.device('cuda')

	@torch.jit.script_method
	def forward(self, x):
		prob = 0.5
		#prob = 0

		p = 1 - prob
		scale = 1.0/p


		drop1 = (scale)*torch.bernoulli(torch.full((1, 896), p)).to(device=self.device)
		drop2 = (scale)*torch.bernoulli(torch.full((1, 512), p)).to(device=self.device)
		drop3 = (scale)*torch.bernoulli(torch.full((1, 256), p)).to(device=self.device)
		drop4 = (scale)*torch.bernoulli(torch.full((1, 128), p)).to(device=self.device)
		drop5 = (scale)*torch.bernoulli(torch.full((1, 64), p)).to(device=self.device)

		out1 = self.fc1(x)
		out1 = torch.mul(out1, drop1)

		out2 = self.fc2(out1)
		out2 = torch.mul(out2, drop2)

		out3 = self.fc3(out2)
		out3 = torch.mul(out3, drop3)

		out4 = self.fc4(out3)
		out4 = torch.mul(out4, drop4)

		out5 = self.fc5(out4)
		out5 = torch.mul(out5, drop5)

		out6 = self.fc6(out5)


		return out6


'''class Pnet_Annotated(torch.jit.ScriptModule):
	__constants__ = ['fc1','fc2','fc3','fc4','fc5','fc6','fc7','device']
	def __init__(self):
		super(Pnet_Annotated, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(398, 512), nn.PReLU())
		self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
		self.fc3 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
		self.fc4 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
		self.fc5 = nn.Sequential(nn.Linear(64, 32), nn.PReLU())
		self.fc6 = nn.Sequential(nn.Linear(32, 7))

		self.device = torch.device('cuda')

	@torch.jit.script_method
	def forward(self, x):
		prob = 0.5
		#prob = 0

		p = 1 - prob
		scale = 1.0/p


		drop1 = (scale)*torch.bernoulli(torch.full((1, 512), p)).to(device=self.device)
		drop2 = (scale)*torch.bernoulli(torch.full((1, 256), p)).to(device=self.device)
		drop3 = (scale)*torch.bernoulli(torch.full((1, 128), p)).to(device=self.device)
		drop4 = (scale)*torch.bernoulli(torch.full((1, 64), p)).to(device=self.device)
		drop5 = (scale)*torch.bernoulli(torch.full((1, 32), p)).to(device=self.device)

		out1 = self.fc1(x)
		out1 = torch.mul(out1, drop1)

		out2 = self.fc2(out1)
		out2 = torch.mul(out2, drop2)

		out3 = self.fc3(out2)
		out3 = torch.mul(out3, drop3)

		out4 = self.fc4(out3)
		out4 = torch.mul(out4, drop4)

		out5 = self.fc5(out4)
		out5 = torch.mul(out5, drop5)

		out6 = self.fc6(out5)


		return out6'''



'''class Pnet_Annotated(torch.jit.ScriptModule):
	__constants__ = ['fc1','fc2','fc3','fc4','fc5','fc6','fc7','device']
	def __init__(self):
		super(Pnet_Annotated, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(398, 896), nn.PReLU())
		self.fc2 = nn.Sequential(nn.Linear(896, 512), nn.PReLU())
		self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
		self.fc4 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
		self.fc5 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
		self.fc6 = nn.Sequential(nn.Linear(64, 32), nn.PReLU())
		self.fc7 = nn.Sequential(nn.Linear(32, 7))

		self.device = torch.device('cuda')

	@torch.jit.script_method
	def forward(self, x):
		prob = 0.5
		#prob = 0

		p = 1 - prob
		scale = 1.0/p


		drop1 = (scale)*torch.bernoulli(torch.full((1, 896), p)).to(device=self.device)
		drop2 = (scale)*torch.bernoulli(torch.full((1, 512), p)).to(device=self.device)
		drop3 = (scale)*torch.bernoulli(torch.full((1, 256), p)).to(device=self.device)
		drop4 = (scale)*torch.bernoulli(torch.full((1, 128), p)).to(device=self.device)
		drop5 = (scale)*torch.bernoulli(torch.full((1, 64), p)).to(device=self.device)
		drop6 = (scale)*torch.bernoulli(torch.full((1, 32), p)).to(device=self.device)

		out1 = self.fc1(x)
		out1 = torch.mul(out1, drop1)

		out2 = self.fc2(out1)
		out2 = torch.mul(out2, drop2)

		out3 = self.fc3(out2)
		out3 = torch.mul(out3, drop3)

		out4 = self.fc4(out3)
		out4 = torch.mul(out4, drop4)

		out5 = self.fc5(out4)
		out5 = torch.mul(out5, drop5)

		out6 = self.fc6(out5)
		out6 = torch.mul(out6, drop6)

		out7 = self.fc7(out6)

		return out7'''


'''class Pnet_Annotated(torch.jit.ScriptModule):
	__constants__ = ['fc1','fc2','fc3','fc4','fc5','fc6','fc7','fc8','device']
	def __init__(self):
		super(Pnet_Annotated, self).__init__()
		self.fc1 = nn.Sequential(nn.Linear(398, 1280), nn.PReLU())
		self.fc2 = nn.Sequential(nn.Linear(1280, 896), nn.PReLU())
		self.fc3 = nn.Sequential(nn.Linear(896, 512), nn.PReLU())
		self.fc4 = nn.Sequential(nn.Linear(512, 384), nn.PReLU())
		self.fc5 = nn.Sequential(nn.Linear(384, 256), nn.PReLU())
		self.fc6 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
		self.fc7 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
		self.fc8 = nn.Sequential(nn.Linear(64, 32),nn.PReLU())
		self.fc9 = nn.Linear(32,7)

		self.device = torch.device('cuda')

	@torch.jit.script_method
	def forward(self, x):
		prob = 0.5

		p = 1 - prob
		scale = 1.0/p

		drop0 = (scale)*torch.bernoulli(torch.full((1, 398), p)).to(device=self.device)
		drop1 = (scale)*torch.bernoulli(torch.full((1, 1280), p)).to(device=self.device)
		drop2 = (scale)*torch.bernoulli(torch.full((1, 896), p)).to(device=self.device)
		drop3 = (scale)*torch.bernoulli(torch.full((1, 512), p)).to(device=self.device)
		drop4 = (scale)*torch.bernoulli(torch.full((1, 384), p)).to(device=self.device)
		drop5 = (scale)*torch.bernoulli(torch.full((1, 256), p)).to(device=self.device)
		drop6 = (scale)*torch.bernoulli(torch.full((1, 128), p)).to(device=self.device)
		drop7 = (scale)*torch.bernoulli(torch.full((1, 64), p)).to(device=self.device)

		out1 = self.fc1(x)
		out1 = torch.mul(out1, drop1)

		out2 = self.fc2(out1)
		out2 = torch.mul(out2, drop2)

		out3 = self.fc3(out2)
		out3 = torch.mul(out3, drop3)

		out4 = self.fc4(out3)
		out4 = torch.mul(out4, drop4)

		out5 = self.fc5(out4)
		out5 = torch.mul(out5, drop5)

		out6 = self.fc6(out5)
		out6 = torch.mul(out6, drop6)

		out7 = self.fc7(out6)
		out7 = torch.mul(out7, drop7)

		out8 = self.fc8(out7)

		out9 = self.fc9(out8)

		return out9'''



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

'''def copypnet(MLP_to_copy, mlp_weights):
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
	MLP_to_copy.state_dict()['fc6.0.weight'].copy_(mlp_weights['fc.15.weight'])
	MLP_to_copy.state_dict()['fc7.0.weight'].copy_(mlp_weights['fc.17.weight'])


	MLP_to_copy.state_dict()['fc1.0.bias'].copy_(mlp_weights['fc.0.bias'])
	MLP_to_copy.state_dict()['fc2.0.bias'].copy_(mlp_weights['fc.3.bias'])
	MLP_to_copy.state_dict()['fc3.0.bias'].copy_(mlp_weights['fc.6.bias'])
	MLP_to_copy.state_dict()['fc4.0.bias'].copy_(mlp_weights['fc.9.bias'])
	MLP_to_copy.state_dict()['fc5.0.bias'].copy_(mlp_weights['fc.12.bias'])
	MLP_to_copy.state_dict()['fc6.0.bias'].copy_(mlp_weights['fc.15.bias'])
	MLP_to_copy.state_dict()['fc7.0.bias'].copy_(mlp_weights['fc.17.bias'])


	MLP_to_copy.state_dict()['fc1.1.weight'].copy_(mlp_weights['fc.1.weight'])
	MLP_to_copy.state_dict()['fc2.1.weight'].copy_(mlp_weights['fc.4.weight'])
	MLP_to_copy.state_dict()['fc3.1.weight'].copy_(mlp_weights['fc.7.weight'])
	MLP_to_copy.state_dict()['fc4.1.weight'].copy_(mlp_weights['fc.10.weight'])
	MLP_to_copy.state_dict()['fc5.1.weight'].copy_(mlp_weights['fc.13.weight'])
	MLP_to_copy.state_dict()['fc6.1.weight'].copy_(mlp_weights['fc.16.weight'])


	return MLP_to_copy'''


'''def copypnet(MLP_to_copy, mlp_weights):
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
	MLP_to_copy.state_dict()['fc6.0.weight'].copy_(mlp_weights['fc.15.weight'])
	MLP_to_copy.state_dict()['fc7.0.weight'].copy_(mlp_weights['fc.18.weight'])
	MLP_to_copy.state_dict()['fc8.0.weight'].copy_(mlp_weights['fc.21.weight'])
	MLP_to_copy.state_dict()['fc9.weight'].copy_(mlp_weights['fc.23.weight'])

	MLP_to_copy.state_dict()['fc1.0.bias'].copy_(mlp_weights['fc.0.bias'])
	MLP_to_copy.state_dict()['fc2.0.bias'].copy_(mlp_weights['fc.3.bias'])
	MLP_to_copy.state_dict()['fc3.0.bias'].copy_(mlp_weights['fc.6.bias'])
	MLP_to_copy.state_dict()['fc4.0.bias'].copy_(mlp_weights['fc.9.bias'])
	MLP_to_copy.state_dict()['fc5.0.bias'].copy_(mlp_weights['fc.12.bias'])
	MLP_to_copy.state_dict()['fc6.0.bias'].copy_(mlp_weights['fc.15.bias'])
	MLP_to_copy.state_dict()['fc7.0.bias'].copy_(mlp_weights['fc.18.bias'])
	MLP_to_copy.state_dict()['fc8.0.bias'].copy_(mlp_weights['fc.21.bias'])
	MLP_to_copy.state_dict()['fc9.bias'].copy_(mlp_weights['fc.23.bias'])

	MLP_to_copy.state_dict()['fc1.1.weight'].copy_(mlp_weights['fc.1.weight'])
	MLP_to_copy.state_dict()['fc2.1.weight'].copy_(mlp_weights['fc.4.weight'])
	MLP_to_copy.state_dict()['fc3.1.weight'].copy_(mlp_weights['fc.7.weight'])
	MLP_to_copy.state_dict()['fc4.1.weight'].copy_(mlp_weights['fc.10.weight'])
	MLP_to_copy.state_dict()['fc5.1.weight'].copy_(mlp_weights['fc.13.weight'])
	MLP_to_copy.state_dict()['fc6.1.weight'].copy_(mlp_weights['fc.16.weight'])
	MLP_to_copy.state_dict()['fc7.1.weight'].copy_(mlp_weights['fc.19.weight'])
	MLP_to_copy.state_dict()['fc8.1.weight'].copy_(mlp_weights['fc.22.weight'])


	return MLP_to_copy'''





def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)


def get_input(prob):
	xt=torch.from_numpy(np.array(prob["rsconfig"],dtype=np.float32)).unsqueeze(0)
	xT=torch.from_numpy(np.array(prob["rgconfig"],dtype=np.float32)).unsqueeze(0)
	c=torch.from_numpy(np.array(prob["one_hot"],dtype=np.float32)).unsqueeze(0)
	voxel=torch.from_numpy(np.array(prob["voxel"],dtype=np.float32)).unsqueeze(0)

	return to_var(xt),to_var(xT),to_var(c),to_var(voxel)



# everything here is the same as above, except we don't need to use the big GEM model, just the regular models
#enet_annotated=VoxelEncoder_Annotated()
#enet_c_annotated=OneHotEncoder_Annotated()
pnet=Pnet()
pnet_annotated=Pnet_Annotated()


device = torch.device('cuda')
#enet_filename = 'models/enet400.pkl'
#enet_c_filename = 'models/enet_c400.pkl'
pnet_filename = 'models/pnet400.pkl'
#pnet_filename = 'models/6_B2_4Best/pnet400.pkl'
pnet_weights = torch.load(pnet_filename)

pnet_to_copy = copypnet(pnet, pnet_weights)
torch.save(pnet_to_copy.state_dict(), 'pnet_no_dropout.pkl')
pnet_annotated.load_state_dict(torch.load('pnet_no_dropout.pkl', map_location=device))
#pnet_annotated.load_state_dict(pnet_to_copy.state_dict(),map_location=device)
#pnet_annotated.cuda()
pnet_annotated.save("mpnet_annotated_gpu_rpp_ohot.pt")



#enet_annotated.load_state_dict(torch.load(enet_filename,map_location='cpu'))
#enet_annotated.save("ctenet_annotated_cpu.pt")

#enet_c_annotated.load_state_dict(torch.load(enet_c_filename,map_location='cpu'))
#enet_c_annotated.save("ctenet_c_annotated_cpu.pt")



### model testing
'''test_scenes=load_test_dataset()
prob=test_scenes["s_0"]["coke_can"]
xt,xT,c,voxel=get_input(prob)

z=enet_annotated(voxel)
cz=enet_c_annotated(c)

z_orig=enet_orig(voxel)
cz_orig=enet_c_orig(c)
print("env")
print(z)
print(z_orig)
print("ohot")
print(cz)
print(cz_orig)
input("==========================")
xt_temp=xt
for i in range (0,10):
	inp1=torch.cat((z,cz,xt,xT),1)
	start1=pnet_orig(inp1)
	print("pnet")
	print(start1)

	inp1=torch.cat((z,cz,xt,xT),1)
	start1_temp=pnet_annotated(inp1)
	#start1_temp=pnet_to_copy(inp1)
	print("pnet_annotated")
	print(start1_temp)
	print("====================================")
	#xt_temp=start1_temp
	#xt=start1'''
'''obs_pc = np.loadtxt('./trainEnv_4_pcd_normalized.csv', dtype=np.float32)
obs = torch.from_numpy(obs_pc)
obs = Variable(obs)
h = encoder(obs)
path_data = np.array([-0.08007369,  0.32780212, -0.01338363,  0.00726194, 0.00430644, -0.00323558,
                   0.18593094,  0.13094018, 0.18499476, 0.3250918, 0.52175426, 0.07388325, -0.49999127, 0.52322733])
path_data = torch.from_numpy(path_data).type(torch.FloatTensor)

test_input = torch.cat((path_data, h.data.cpu())).cuda()  # for MPNet1.0
test_input = Variable(test_input)

for i in range(5):
	test_output = MLP_to_copy(test_input)
	test_output_save = MLP(test_input)
	print("output %d: " % i	)
	print(test_output.data)
	print(test_output_save.data)'''

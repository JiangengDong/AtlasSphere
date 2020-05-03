import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
'''resnet34 = models.resnet34(pretrained=True)
#modules=list(resnet50.children())[:-3]
modules=list(resnet34.children())[:-1]
resnet34=nn.Sequential(*modules)
for p in resnet34.parameters():
    p.requires_grad = False

example = torch.rand(1, 3, 224, 224)
output=resnet34(example)

print(output.size())'''

class ConstraintEncoder(nn.Module):
	def __init__(self):
		super(ConstraintEncoder, self).__init__()
		self.image_size = 300
		self.num_channels = 3
		self.embed_dim = 4096
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16

		self.netD_1 = nn.Sequential(
			# input is (nc) x 300 x 300
			nn.Conv2d(self.num_channels, 64, 8, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.BatchNorm2d(64),
			# state size. 64 x 148 x 148
			nn.Conv2d(64, 32, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.BatchNorm2d(32),
			# state size. 32 x 37 x 37
			nn.Conv2d(32, 32, 3, 4, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.BatchNorm2d(32), 			# state size. 32 x 5 x 5
		)
		self.projection = nn.Sequential(
			nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.BatchNorm1d(num_features=self.projected_embed_dim)
		)

		self.netD_2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4 # fully connected
			nn.Linear(in_features=self.projected_embed_dim+ 64*5*5, out_features=256),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.BatchNorm1d(num_features=256)
		)

	def forward(self, inp_view_1, inp_view_2, embed):
		x_1 = self.netD_1(inp_view_1)
		x_2 = self.netD_1(inp_view_2)
		x_concate=torch.cat([x_1,x_2],1).view(-1, 64*5*5) # B x 64 x 5 x 5  ---> B x 64*5*5 --> B x 1600
		txt_embd= self.projection(embed) # B x 128 
		features= torch.cat([x_concate,txt_embd],1) # B x 1728
		out = self.netD_2(features) # B x 256 
		return out


'''class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv3d(in_channels=1, out_channels=32, kernel_size=[5,5,5], stride=[2,2,2]),
			nn.PReLU(),
			nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1]),
			nn.PReLU(),
			nn.MaxPool3d(kernel_size=[2,2,2])
		)
		x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
		first_fc_in_features = 1
		for n in x.size()[1:]:
			first_fc_in_features *= n
		self.head = nn.Sequential(
			nn.Linear(first_fc_in_features, 128),
			nn.PReLU(),
			nn.Linear(128, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x'''

#2
'''class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv3d(in_channels=1, out_channels=32, kernel_size=[5,5,5], stride=[2,2,2]),
			nn.PReLU(),
			nn.Conv3d(in_channels=32, out_channels=64, kernel_size=[3,3,3], stride=[1,1,1]),
			nn.PReLU(),
			nn.MaxPool3d(kernel_size=[2,2,2])
		)
		x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
		first_fc_in_features = 1
		for n in x.size()[1:]:
			first_fc_in_features *= n
		self.head = nn.Sequential(
			nn.Linear(first_fc_in_features, 256),
			nn.PReLU(),
			nn.Linear(256, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x'''

#3
'''class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv3d(in_channels=1, out_channels=256, kernel_size=[5,5,5], stride=[2,2,2]),
			nn.PReLU(),
			nn.Conv3d(in_channels=256, out_channels=128, kernel_size=[3,3,3], stride=[1,1,1]),
			nn.PReLU(),
			nn.MaxPool3d(kernel_size=[2,2,2])
		)
		x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
		first_fc_in_features = 1
		for n in x.size()[1:]:
			first_fc_in_features *= n
		self.head = nn.Sequential(
			nn.Linear(first_fc_in_features, 512),
			nn.PReLU(),
			nn.Linear(512, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x'''

#4
'''class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=33, out_channels=256, kernel_size=[5,5], stride=[2,2]),
			nn.PReLU(),
			nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[3,3], stride=[1,1]),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size=[2,2])
		)
		x = self.encoder(torch.autograd.Variable(torch.rand([1, 33] + input_size)))
		first_fc_in_features = 1
		for n in x.size()[1:]:
			first_fc_in_features *= n
		self.head = nn.Sequential(
			nn.Linear(first_fc_in_features, 512),
			nn.PReLU(),
			nn.Linear(512, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x'''

#5
'''class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=33, out_channels=128, kernel_size=[5,5], stride=[2,2]),
			nn.PReLU(),
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[3,3], stride=[1,1]),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size=[2,2])
		)
		x = self.encoder(torch.autograd.Variable(torch.rand([1, 33] + input_size)))
		first_fc_in_features = 1
		for n in x.size()[1:]:
			first_fc_in_features *= n
		self.head = nn.Sequential(
			nn.Linear(first_fc_in_features, 256),
			nn.PReLU(),
			nn.Linear(256, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x'''

#6Best

class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[5,5], stride=[2,2]),
			nn.PReLU(),
			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3,3], stride=[1,1]),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size=[2,2])
		)
		x = self.encoder(torch.autograd.Variable(torch.rand([1, 32] + input_size)))
		first_fc_in_features = 1
		for n in x.size()[1:]:
			first_fc_in_features *= n
		self.head = nn.Sequential(
			nn.Linear(first_fc_in_features, 256),
			nn.PReLU(),
			nn.Linear(256, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x





#7
'''class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=33, out_channels=32, kernel_size=[5,5], stride=[2,2]),
			nn.PReLU(),
			nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3,3], stride=[1,1]),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size=[2,2])
		)
		x = self.encoder(torch.autograd.Variable(torch.rand([1, 33] + input_size)))
		first_fc_in_features = 1
		for n in x.size()[1:]:
			first_fc_in_features *= n
		self.head = nn.Sequential(
			nn.Linear(first_fc_in_features, 256),
			nn.PReLU(),
			nn.Linear(256, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x'''


#8
'''class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv3d(in_channels=1, out_channels=32, kernel_size=[5,5,5], stride=[2,2,2]),
			nn.PReLU(),
			nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3,3,3], stride=[1,1,1]),
			nn.PReLU(),
			nn.MaxPool3d(kernel_size=[2,2,2])
		)
		x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
		first_fc_in_features = 1
		for n in x.size()[1:]:
			first_fc_in_features *= n
		self.head = nn.Sequential(
			nn.Linear(first_fc_in_features, 256),
			nn.PReLU(),
			nn.Linear(256, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x'''

#9
'''class VoxelEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(VoxelEncoder, self).__init__()
		input_size = [input_size, input_size]
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=33, out_channels=128, kernel_size=[5,5], stride=[2,2]),
			nn.PReLU(),
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[3,3], stride=[1,1]),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size=[2,2])
		)
		x = self.encoder(torch.autograd.Variable(torch.rand([1, 33] + input_size)))
		first_fc_in_features = 1
		for n in x.size()[1:]:
			first_fc_in_features *= n
		self.head = nn.Sequential(
			nn.Linear(first_fc_in_features, 512),
			nn.PReLU(),
			nn.Linear(512, output_size)
		)
	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)
		x = self.head(x)
		return x'''

class ConstraintDecoder(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers):
		"""Set the hyper-parameters and build the layers."""
		super(DecoderRNN, self).__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, output_size)
		#self.init_weights()
	    
	#def init_weights(self):
		#"""Initialize weights."""
		#self.linear.weight.data.uniform_(-0.1, 0.1)
		#self.linear.bias.data.fill_(0)
        
	def forward(self, features):
		"""Decode image feature vectors and generates captions."""
		hiddens, _ = self.lstm(features)
		outputs = self.linear(hiddens[0])
		return outputs

	def forward(self, features, paths, lengths):
		"""Decode image feature vectors and generates captions."""
		inputs = torch.cat((features.unsqueeze(1), paths), 1)
		packed = pack_padded_sequence(inputs, lengths, batch_first=True) 
		hiddens, _ = self.lstm(packed)
		outputs = self.linear(hiddens[0])
		return outputs
	def sample(self, features, states=None):
		"""Samples trajectory points for given env features (Greedy search)."""
		inputs = features.unsqueeze(1)
		hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 		
		outputs = self.linear(hiddens[0])            # (batch_size, vocab_size)
		#print outputs
		return outputs


#1
'''class PNet(nn.Module): #didn't work
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
		nn.Linear(1280, 896), nn.PReLU(), nn.Dropout(),
		nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
		nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32), nn.PReLU(),
		nn.Linear(32, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''


#2
'''class PNet(nn.Module):
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
		nn.Linear(1280, 896), nn.PReLU(), nn.Dropout(),
		nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
		#nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
		#nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 64), nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32), nn.PReLU(),
		nn.Linear(32, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''

#3*
'''class PNet(nn.Module):
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 64), nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32), nn.PReLU(),
		nn.Linear(32, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''

#4Best
'''class PNet(nn.Module): #didn't work
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 896), nn.PReLU(), nn.Dropout(),
		nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32), nn.PReLU(),
		nn.Linear(32, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''

#5
'''class PNet(nn.Module):
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 896), nn.PReLU(), nn.Dropout(),
		nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 64), nn.PReLU(),
		nn.Linear(64, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''


#6
'''class PNet(nn.Module): #didn't work
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
		nn.Linear(1280, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32), nn.PReLU(),
		nn.Linear(32, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''

#7

'''class PNet(nn.Module): #didn't work
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32), nn.PReLU(),
		nn.Linear(32, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''

#8
class PNet(nn.Module): #didn't work
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 896), nn.PReLU(), nn.Dropout(),
		nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64), nn.PReLU(),
		nn.Linear(64, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out



#A1
class Enet_im(nn.Module):
	def __init__(self, input_size, output_size):
		super(Enet_im, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 512), nn.ReLU(),
		nn.Linear(512, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out
#B1
'''class Enet_constraint(nn.Module):
	def __init__(self, input_size, output_size):
		super(Enet_constraint, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 64), nn.PReLU(),
		nn.Linear(64, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''
#B2
'''class Enet_constraint(nn.Module):
	def __init__(self, input_size, output_size):
		super(Enet_constraint, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 128), nn.PReLU(),
		nn.Linear(128, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''


#B3
'''class Enet_constraint(nn.Module):
	def __init__(self, input_size, output_size):
		super(Enet_constraint, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 256), nn.PReLU(),
		nn.Linear(256, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out'''

#B3
class Enet_constraint(nn.Module):
	def __init__(self, input_size, output_size):
		super(Enet_constraint, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 128), nn.PReLU(),
		nn.Linear(128, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out

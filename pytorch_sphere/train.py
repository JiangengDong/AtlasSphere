import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable 
import math
from data_loader import load_samples
from models import VoxelEncoder, PNet,Enet_constraint
from torch.optim.lr_scheduler import StepLR

inp_states,inp_vox,out_states,inp_ohot=load_samples()


def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def get_input(i,bs):
	
	#print("i:"+str(i))
	#print("bs:"+str(bs))
	#print("size:"+str(len(inp_states)))
	#print("size:"+str(len(inp_vox)))
	#print("size:"+str(len(out_states)))
	if i+bs<len(inp_states):
		#if i!=384:
		#print(np.shape(inp_vox[i:i+bs]))
		#print(np.size(inp_vox[i:i+bs]))			
		bi_states=np.array(inp_states[i:i+bs],dtype=np.float32)
		bi_voxel=np.array(inp_vox[i:i+bs],dtype=np.float32)
		bo_states=np.array(out_states[i:i+bs],dtype=np.float32)
		bi_ohot=np.array(inp_ohot[i:i+bs],dtype=np.float32)
	else:
		bi_states=np.array(inp_states[i:],dtype=np.float32)
		bi_voxel=np.array(inp_vox[i:],dtype=np.float32)
		bo_states=np.array(out_states[i:],dtype=np.float32)
		bi_ohot=np.array(inp_ohot[i:],dtype=np.float32)


	#return torch.from_numpy(bi_states),torch.from_numpy(bi_voxel).unsqueeze(1),torch.from_numpy(bo_states),torch.from_numpy(bi_ohot)
	return torch.from_numpy(bi_states),torch.from_numpy(bi_voxel),torch.from_numpy(bo_states),torch.from_numpy(bi_ohot)


    
def main(args):
	# Create model directory
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	
	# Build the models
	enet = VoxelEncoder(args.insz_enet, args.outsz_enet)
	pnet = PNet(args.insz_pnet, args.outsz_pnet)
	#enet_c=Enet_constraint(5,32)
	enet_c=Enet_constraint(4096,128)
	print(pnet)
	print(enet)
	print(enet_c)
    
	if torch.cuda.is_available():
		enet=enet.cuda()
		pnet=pnet.cuda()
		enet_c=enet_c.cuda()
		print("available")
	else:
		print("not available")

	# Loss and Optimizer
	criterion = nn.MSELoss()
	params=list(enet.parameters())+list(enet_c.parameters())+list(pnet.parameters()) 
	optimizer = torch.optim.Adagrad(params) 
	#scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
	# Train the Models
	total_loss=[]
	sm=100 # start saving models after 100 epochs
	for epoch in range(args.num_epochs):
		print "epoch" + str(epoch)
		avg_loss=0
		for i in range (0,len(inp_states),args.batch_size):
			# Forward, Backward and Optimize
			pnet.zero_grad()
			enet.zero_grad()
			enet_c.zero_grad()			
			bi_states,bi_voxel,bo_states,bi_ohot=get_input(i,args.batch_size)
			bi_states=to_var(bi_states)
			bi_voxel=to_var(bi_voxel)
			bo_states=to_var(bo_states)

			bi_ohot=to_var(bi_ohot)
			bo_ohot = enet_c.forward(bi_ohot)

			bo_voxel = enet.forward(bi_voxel)


			#bi_pnet=torch.cat((bo_voxel,bi_states),1)
			bi_pnet=torch.cat((bo_voxel,bo_ohot,bi_states),1)

			bo_pnet=pnet.forward(bi_pnet)
			loss = criterion(bo_pnet,bo_states)
			avg_loss=avg_loss+loss.item()
			loss.backward()
			optimizer.step()
		#scheduler.step()
		print "--average loss:"
		print avg_loss/(len(inp_states)/args.batch_size)
		total_loss.append(avg_loss/(len(inp_states)/args.batch_size))
		# Save the models
		if epoch%args.save_step==0:
			enet_path='enet'+str(epoch)+'.pkl'
			torch.save(enet.state_dict(),os.path.join(args.model_path,enet_path))
			pnet_path='pnet'+str(epoch)+'.pkl'
			torch.save(pnet.state_dict(),os.path.join(args.model_path,pnet_path))
			pnet_path='enet_c'+str(epoch)+'.pkl'
			torch.save(enet_c.state_dict(),os.path.join(args.model_path,pnet_path))


	loss_path='total_loss.dat'
	torch.save(total_loss,os.path.join(args.model_path,loss_path))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
	parser.add_argument('--save_step', type=int , default=10,help='step size for saving trained models')

	# Model parameters
	parser.add_argument('--insz_enet', type=int , default=32, help='dimension of ENets input vector')
	parser.add_argument('--outsz_enet', type=int , default=256, help='dimension of ENets output vector')

	parser.add_argument('--insz_pnet', type=int , default=398, help='dimension of PNets input vector')#83
	parser.add_argument('--outsz_pnet', type=int , default=7, help='dimension of PNets output vector')

	parser.add_argument('--num_epochs', type=int, default=401)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--lr', type=float, default=0.0001)
	args = parser.parse_args()
	print(args)
	main(args)

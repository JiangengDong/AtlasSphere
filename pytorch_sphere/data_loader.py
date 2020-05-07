import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math

import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np


'''preprocess = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
resnet50 = models.resnet50(pretrained=True)
modules=list(resnet50.children())[:-2]
resnet50=nn.Sequential(*modules)
for p in resnet50.parameters():
	p.requires_grad = False'''

def get_voxels(obj_order, scene_voxel, obj_voxel):
	voxels=[]
	voxels.append(scene_voxel)
	for i in range(0, len(obj_order)):
		voxel=np.array(obj_voxel[obj_order[i]],dtype=np.float32)
		if np.shape(voxel)[0]==35:
			voxel = voxel[:-2,:-2,:-2]
		voxels.append(voxel)
		#voxels.append(obj_voxel[obj_order[i]])
	return voxels

def get_voxels_door(obj_order, voxel_data):
	voxels=[]
	voxel=np.array(voxel_data["full"],dtype=np.float32)
	voxels.append(voxel)
	for i in range(0, len(obj_order)):
		voxel=np.array(voxel_data[obj_order[i]],dtype=np.float32)
		if np.shape(voxel)[0]==35:
			voxel = voxel[:-3,:-3,:-3]
		voxels.append(voxel)
		#voxels.append(obj_voxel[obj_order[i]])
	return voxels

def preprocess_image(img):
	img=Image.fromarray(img)
	img_tensor = preprocess(img)
	img_batch = img_tensor.unsqueeze(0)
	features=resnet50(img_batch)
	features=features.view(-1, 2048*7*7)
	features=features.squeeze(0)
	return features.numpy()	 

def get_images(obj_order, rgbs):
	images=[]
	img=rgbs["full"]
	features=preprocess_image(img)
	images.append(features)
	for i in range(0, len(obj_order)):
		img=rgbs[obj_order[i]]
		features=preprocess_image(img)
		images.append(features)
	return images

def get_features(obj_order, resnet_f):
	images=[]
	features=resnet_f["full"]
	images.append(features)
	for i in range(0, len(obj_order)):
		features=resnet_f[obj_order[i]]
		images.append(features)
	return images
#joint_ranges = np.array([3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67]) ## Anthony
joint_ranges = np.array([6.1083,2.668,3.4033,3.194,6.118,3.6647,6.118])
#path_np.append(np.multiply(state.numpy(), joint_ranges))
#new_path = np.divide(new_path, self.joint_range) 
	

#N=number of environments; NP=Number of Paths
def load_dataset_door(env=89,scene=30):
	esc_dict = pickle.load(open( "/media/ahmed/DATA/comps-code/bax_door_data/data/esp_dict_door_0_88_30.p", "rb" ))

	esc_ohot = pickle.load(open( "text_embed/InferSent/words_embeddings_door.p", "rb" ))
	obj=['juice', 'fuze_bottle', 'coke_can', 'door', 'mugblack', 'plasticmug', 'pitcher']
	s=0
	scenes={}
	tp=0
	for i in range(0,env):
		env_no="env_"+str(i)
		path_voxel="/media/ahmed/DATA/comps-code/bax_door_data/esc_voxel"+str(i)+"_30.pkl"
		esc_voxel = pickle.load(open(path_voxel,"rb" ))
		#print(env_no)
		if env_no in esc_dict.keys():
			for j in range (0,scene):
				s_no="s_"+str(j)
				if s_no in esc_dict[env_no].keys():
					obj_order=esc_dict[env_no][s_no]["obj_order"]
					
					voxel_data=esc_voxel[env_no][s_no]
					voxels=get_voxels_door(obj_order,voxel_data)

					for k in range(0,len(obj)):
						voxel=voxels[k]
						traj=[]
						voxel_traj=[]
						traj_target=[]
						ohot=[]

						txt_name=obj_order[k]+"_pp"
						txt_embed=esc_ohot[txt_name]
						path_pp=None
						if obj_order[k]=="door":
							path_pp=esc_dict[env_no][s_no]["initial"][obj_order[k]]["path_pick_place"]
						else:
							path_pp=esc_dict[env_no][s_no][obj_order[k]]["path_pick_place"]	
						path_pp=np.divide(path_pp,joint_ranges)
						tp=tp+1
						c_goal=path_pp[len(path_pp)-1]
						for p in range(0, len(path_pp)-1):
							voxel_traj.append(voxel)
							c_current=path_pp[p]
							samp=np.concatenate([c_current,c_goal],0)
							c_next=path_pp[p+1]
							traj.append(samp)
							traj_target.append(c_next)
							ohot.append(txt_embed)


						#reach combined
						txt_name=obj_order[k]+"_reach"
						txt_embed=esc_ohot[txt_name]
						path_reach=None
						if obj_order[k]=="door":
							path_reach=esc_dict[env_no][s_no]["initial"][obj_order[k]]["path_reach"]
						else:
							path_reach=esc_dict[env_no][s_no][obj_order[k]]["path_reach"]	
	
						path_reach=np.divide(path_reach,joint_ranges)
						tp=tp+1
						c_goal=path_reach[len(path_reach)-1]
						for p in range(0, len(path_reach)-1):
							voxel_traj.append(voxel)
							c_current=path_reach[p]
							samp=np.concatenate([c_current,c_goal],0)
							c_next=path_reach[p+1]
							traj.append(samp)
							traj_target.append(c_next)
							ohot.append(txt_embed)


	
						sc="s_"+str(s)
						if sc in scenes.keys():
							scenes[sc].update({obj_order[k]:{"input":traj,"output":traj_target,"voxel":voxel_traj,"one_hots":ohot}, "env_no":env_no,"s_no":s_no, "obj_order":obj_order})
						else:
							scenes[sc]={obj_order[k]:{"input":traj,"output":traj_target,"voxel":voxel_traj,"one_hots":ohot}, "env_no":env_no,"s_no":s_no, "obj_order":obj_order}
					s=s+1

	return scenes, tp


def load_dataset(env=20,scene=110):
	esc_dict = pickle.load(open( "../examples/python_baxter/data/esp_dict0_19_120.p", "rb" ))
	esc_voxel = pickle.load(open( "../examples/python_baxter/voxels/escf_voxel0_19_120.p", "rb" ))
	esco_voxel = pickle.load(open( "../examples/python_baxter/voxels/esco_voxel0_19_120.p", "rb" ))

	esc_ohot = pickle.load(open( "text_embed/InferSent/words_embeddings2.p", "rb" ))
	obj=['juice', 'fuze_bottle', 'coke_can', 'plasticmug', 'teakettle']
	s=0
	scenes={}
	tp=0
	for i in range(0,env):
		env_no="env_"+str(i)
		#print(env_no)
		if env_no in esc_dict.keys():
			for j in range (0,scene):
				s_no="s_"+str(j)
				if s_no in esc_dict[env_no].keys():
					if (env_no!="env_6" and s_no!="s_21") and (env_no!="env_16" and s_no!="s_81") : 
						#print(s_no)
						obj_order=esc_dict[env_no][s_no]["obj_order"]
						
						if "voxel" in esc_voxel[env_no][s_no].keys():
							voxel=esc_voxel[env_no][s_no]["voxel"]
						else:
							voxel=esc_voxel[env_no][s_no]["full_scene"]
						voxels=get_voxels(obj_order,voxel,esco_voxel[env_no][s_no])

						#for k in range(0,1):
						for k in range(0,len(obj)):
							voxel=voxels[k]
							#print(np.shape(voxel))
							traj=[]
							voxel_traj=[]
							traj_target=[]
							ohot=[]
							one_hot=np.zeros(len(obj),dtype=np.float32)
							indx=obj.index(obj_order[k])
							one_hot[indx]=1.0
							#txt_embed=esc_ohot[obj_order[k]]

							txt_name=obj_order[k]+"_pp"
							txt_embed=esc_ohot[txt_name]	
							#path_pp=esc_dict[env_no][s_no][obj_order[k]]["path_pick_place"]path_reach
							print(env_no)
							print(s_no)
							#path_pp=np.divide(esc_dict[env_no][s_no][obj_order[k]]["path_reach"],joint_ranges)
							path_pp=np.divide(esc_dict[env_no][s_no][obj_order[k]]["path_pick_place"],joint_ranges)
							tp=tp+1
							#sample=np.zeros(14,dtype=np.float32)
							c_goal=path_pp[len(path_pp)-1]
							for p in range(0, len(path_pp)-1):
								voxel_traj.append(voxel)
								c_current=path_pp[p]
								#samp=np.concatenate([one_hot,c_current,c_goal],0)
								samp=np.concatenate([c_current,c_goal],0)
								c_next=path_pp[p+1]
								traj.append(samp)
								traj_target.append(c_next)
								#ohot.append(one_hot)
								ohot.append(txt_embed)


							#reach combined
							txt_name=obj_order[k]+"_reach"
							txt_embed=esc_ohot[txt_name]	
							path_pp=np.divide(esc_dict[env_no][s_no][obj_order[k]]["path_reach"],joint_ranges)
							tp=tp+1
							c_goal=path_pp[len(path_pp)-1]
							for p in range(0, len(path_pp)-1):
								voxel_traj.append(voxel)
								c_current=path_pp[p]
								samp=np.concatenate([c_current,c_goal],0)
								c_next=path_pp[p+1]
								traj.append(samp)
								traj_target.append(c_next)
								ohot.append(txt_embed)


		
							#bidirectional training
							'''c_goal=path_pp[0]
							for p in range(len(path_pp)-1,0,-1):
								voxel_traj.append(voxel)
								c_current=path_pp[p]
								#samp=np.concatenate([one_hot,c_current,c_goal],0)
								samp=np.concatenate([c_current,c_goal],0)
								c_next=path_pp[p-1]
								traj.append(samp)
								traj_target.append(c_next)
								ohot.append(one_hot)'''

							sc="s_"+str(s)
							if sc in scenes.keys():
								scenes[sc].update({obj_order[k]:{"input":traj,"output":traj_target,"voxel":voxel_traj,"one_hots":ohot}, "env_no":env_no,"s_no":s_no, "obj_order":obj_order})
							else:
								scenes[sc]={obj_order[k]:{"input":traj,"output":traj_target,"voxel":voxel_traj,"one_hots":ohot}, "env_no":env_no,"s_no":s_no, "obj_order":obj_order}
						s=s+1

	return scenes, tp


'''def load_resnet():
	resnet50 = models.resnet50(pretrained=True)
	modules=list(resnet50.children())[:-2]
	resnet50=nn.Sequential(*modules)
	for p in resnet50.parameters():
		p.requires_grad = False

	return resnet50	


def load_rgb_dataset(env=20,scene=50):
	esc_dict = pickle.load(open( "../examples/python_baxter/esp_dict0_19.p", "rb" ))
	esc_img = pickle.load(open("../examples/python_baxter/voxels/esimg_dict0_19.pkl", "rb"))
	obj=['juice', 'fuze_bottle', 'coke_can', 'plasticmug', 'teakettle']
	s=0
	scenes={}
	tp=0
	for i in range(0,env):
		env_no="env_"+str(i)
		#print(env_no)
		if env_no in esc_dict.keys():
			for j in range (0,scene):
				s_no="s_"+str(j)
				if s_no in esc_dict[env_no].keys():
					if env_no!="env_6" and s_no!="s_21": 
						#print(s_no)
						obj_order=esc_dict[env_no][s_no]["obj_order"]
						features=get_images(obj_order,esc_img[env_no][s_no])
						for k in range(0,len(obj)):
							feature=features[k]
							#print(np.shape(feature))
							#input()
							traj=[]
							feature_traj=[]
							traj_target=[]
							ohot=[]
							one_hot=np.zeros(len(obj),dtype=np.float32)
							indx=obj.index(obj_order[k])
							#print(indx)
							#print(obj_order)
							one_hot[indx]=1.0	
							#path_pp=esc_dict[env_no][s_no][obj_order[k]]["path_pick_place"]path_reach
							path_pp=np.divide(esc_dict[env_no][s_no][obj_order[k]]["path_reach"],joint_ranges)
							tp=tp+1
							#sample=np.zeros(14,dtype=np.float32)
							c_goal=path_pp[len(path_pp)-1]
							for p in range(0, len(path_pp)-1):
								feature_traj.append(feature)
								ohot.append(one_hot)
								c_current=path_pp[p]
								samp=np.concatenate([c_current,c_goal],0)
								c_next=path_pp[p+1]
								traj.append(samp)
								traj_target.append(c_next)

							#bidirectional training
							c_goal=path_pp[0]
							for p in range(len(path_pp)-1,0,-1):
								feature_traj.append(feature)
								ohot.append(one_hot)
								c_current=path_pp[p]
								samp=np.concatenate([c_current,c_goal],0)
								c_next=path_pp[p-1]
								traj.append(samp)
								traj_target.append(c_next)

							sc="s_"+str(s)
							if sc in scenes.keys():
								scenes[sc].update({obj_order[k]:{"input":traj,"output":traj_target,"features":feature_traj,"one_hots":ohot}, "env_no":env_no,"s_no":s_no, "obj_order":obj_order})
							else:
								scenes[sc]={obj_order[k]:{"input":traj,"output":traj_target,"features":feature_traj,"one_hots":ohot}, "env_no":env_no,"s_no":s_no, "obj_order":obj_order}
						s=s+1

	return scenes, tp



		
def load_rgb_batches(names_batch,esc_dict):

	traj=[]
	feature_traj=[]
	traj_target=[]
	ohot=[]

	for i in range(len(names_batch)):
		env,sc=names_batch[i]

		env_no="env_"+str(env)
		s_no="s_"+str(sc)
		scene_no=env_no+"_"+s_no+"_imgs.p"
		paths="/media/ahmed/DATA/comps-code/data_5_12_19/resnet_features/"+env_no+"/e_"+str(env)+"_s_"+str(sc)+"_resnet.p"
		esc_img = pickle.load(open(paths, "rb"))
		obj=['juice', 'fuze_bottle', 'coke_can', 'plasticmug', 'teakettle']
		s=0
		scenes={}
		tp=0

		obj_order=esc_dict[env_no][s_no]["obj_order"]
		features=get_features(obj_order,esc_img)
		#for k in range(0,len(obj)):
		for k in range(0,1):
			if "path_reach" in esc_dict[env_no][s_no][obj_order[k]].keys():
				feature=features[k]
				one_hot=np.zeros(len(obj),dtype=np.float32)
				indx=obj.index(obj_order[k])
				one_hot[indx]=1.0	
				path_pp=np.divide(esc_dict[env_no][s_no][obj_order[k]]["path_reach"],joint_ranges)
				c_goal=path_pp[len(path_pp)-1]
				for p in range(0, len(path_pp)-1):
					feature_traj.append(feature)
					ohot.append(one_hot)
					c_current=path_pp[p]
					samp=np.concatenate([c_current,c_goal],0)
					c_next=path_pp[p+1]
					traj.append(samp)
					traj_target.append(c_next)

				c_goal=path_pp[0]
				for p in range(len(path_pp)-1,0,-1):
					feature_traj.append(feature)
					ohot.append(one_hot)
					c_current=path_pp[p]
					samp=np.concatenate([c_current,c_goal],0)
					c_next=path_pp[p-1]
					traj.append(samp)
					traj_target.append(c_next)


	traj=np.array(traj,dtype=np.float32)
	feature_traj=np.array(feature_traj,dtype=np.float32)
	traj_target=np.array(traj_target,dtype=np.float32)
	ohot=np.array(ohot,dtype=np.float32)

	return torch.from_numpy(traj),torch.from_numpy(feature_traj),torch.from_numpy(traj_target),torch.from_numpy(ohot)


def load_img_samples_new(i,bs,names,esc_dict):
	names_batch=[]
	if i+bs<len(names):
		names_batch=names[i:i+bs]
	else:
		names_batch=names[i:]

	traj,feature_traj,traj_target,ohot=load_rgb_batches(names_batch,esc_dict)
	return traj,feature_traj,traj_target,ohot



def load_img_samples(env=20,scene=50):
	scenes,tp=load_rgb_dataset(env,scene)
	print("total_trajectories:"+str(tp))
	sc=scenes.keys()
	inp_traj=[]
	out_traj=[]
	inp_features=[]
	inp_ohot=[]
	for i in range(0,len(sc)):
		s="s_"+str(i)
		#scene=scenes[sc[i]]
		scene=scenes[s]
		obj_order=scene["obj_order"]
		for j in range(0,len(obj_order)):
			inp_traj+=scene[obj_order[j]]["input"]
			out_traj+=scene[obj_order[j]]["output"]
			inp_features+=scene[obj_order[j]]["features"]
			inp_ohot+=scene[obj_order[j]]["one_hots"]

	data=zip(inp_traj,inp_features,out_traj,inp_ohot)
	random.shuffle(data)	
	inp_traj,inp_features,out_traj,inp_ohot=zip(*data)
	return inp_traj,inp_features,out_traj,inp_ohot'''

def load_samples(env=19,scene=110):
	#scenes,tp=load_dataset(env,scene)
	scenes,tp=load_dataset_door(65,27)
	print("total_trajectories:"+str(tp))
	sc=scenes.keys()
	inp_traj=[]
	out_traj=[]
	inp_voxl=[]
	inp_ohot=[]
	for i in range(0,len(sc)):
		s="s_"+str(i)
		scene=scenes[s]
		obj_order=scene["obj_order"]
		for j in range(0,len(obj_order)):
			inp_traj+=scene[obj_order[j]]["input"]
			out_traj+=scene[obj_order[j]]["output"]
			inp_voxl+=scene[obj_order[j]]["voxel"]
			inp_ohot+=scene[obj_order[j]]["one_hots"]

	return inp_traj,inp_voxl,out_traj,inp_ohot
			
#N=number of environments; NP=Number of Paths
'''def load_dataset2(env=1,scene=1):
	esc_dict = pickle.load(open( "../examples/python_baxter/esp_dict0_19.p", "rb" ))
	esc_voxel = pickle.load(open( "../examples/python_baxter/voxels/full_scene/esc_voxel0_6.p", "rb" ))
	esco_voxel = pickle.load(open( "../examples/python_baxter/voxels/esco_voxel0.p", "rb" ))
	obj=['juice', 'fuze_bottle', 'coke_can', 'plasticmug', 'teakettle']
	s=0
	traj=[]
	voxel_traj=[]
	traj_target=[]			
	for i in range(0,env):
		env_no="env_"+str(i)
		if env_no in esc_dict.keys():
			for j in range (0,scene):
				s_no="s_"+str(j)
				if s_no in esc_dict[env_no].keys():
					obj_order=esc_dict[env_no][s_no]["obj_order"]
					voxels=get_voxels(obj_order,esc_voxel[env_no][s_no]["voxel"],esco_voxel[env_no][s_no])
					for k in range(0,len(obj)):
						voxel=voxels[k]
						one_hot=np.zeros(len(obj),dtype=np.float32)
						indx=obj.index(obj_order[k])
						one_hot[indx]=1.0	
						path_pp=esc_dict[env_no][s_no][obj_order[k]]["path_pick_place"]
						sample=np.zeros(14,dtype=np.float32)
						c_goal=path_pp[len(path_pp)-1]
						for p in range(0, len(path_pp)-1):
							voxel_traj.append(voxel)
							c_current=path_pp[p]
							samp=np.concatenate([one_hot,c_current,c_goal],0)
							c_next=path_pp[p+1]
							traj.append(samp)
							traj_target.append(c_next)


	return scenes'''


def load_test_dataset(test_type=2):
	if test_type==0: #see env
		s_env=0
		e_env=19
		s_sc=50
		e_sc=60
	elif test_type==1: #see env
		s_env=0
		e_env=120
		s_sc=110
		e_sc=120
	else: #see env
		s_env=0
		e_env=1
		s_sc=0
		e_sc=1

	esc_dict = pickle.load(open( "../examples/python_baxter/data/esp_dict0_19_120.p", "rb" ))
	esc_voxel = pickle.load(open( "../examples/python_baxter/voxels/escf_voxel0_19_120.p", "rb" ))
	esco_voxel = pickle.load(open( "../examples/python_baxter/voxels/esco_voxel0_19_120.p", "rb" ))

	obj=['juice', 'fuze_bottle', 'coke_can', 'plasticmug', 'teakettle']
	s=0
	scenes={}
	for i in range(s_env,e_env):
		env_no="env_"+str(i)
		if env_no in esc_dict.keys():
			for j in range (s_sc,e_sc):
				s_no="s_"+str(j)
				if s_no in esc_dict[env_no].keys():
					if env_no!="env_6" and s_no!="s_21": 
						obj_order=esc_dict[env_no][s_no]["obj_order"]
						if "voxel" in esc_voxel[env_no][s_no].keys():
							voxel=esc_voxel[env_no][s_no]["voxel"]
						else:
							voxel=esc_voxel[env_no][s_no]["full_scene"]
						#voxels=get_voxels(obj_order,esc_voxel[env_no][s_no]["voxel"],esco_voxel[env_no][s_no])
						voxels=get_voxels(obj_order,voxel,esco_voxel[env_no][s_no])

						#for k in range(0,1):
						for k in range(0,len(obj)):
							voxel=voxels[k]
							one_hot=np.zeros(len(obj),dtype=np.float32)
							indx=obj.index(obj_order[k])
							one_hot[indx]=1.0	
							#path_pp=esc_dict[env_no][s_no][obj_order[k]]["path_pick_place"]
							#path_pp=esc_dict[env_no][s_no][obj_order[k]]["path_reach"]
							if "path_reach" in esc_dict[env_no][s_no][obj_order[k]].keys():
								path_pp=np.divide(esc_dict[env_no][s_no][obj_order[k]]["path_reach"],joint_ranges)
								#sample=np.zeros(14,dtype=np.float32)
								c_goal=path_pp[len(path_pp)-1]
								c_start=path_pp[0]
								#samp=np.concatenate([one_hot,c_start,c_goal],0)
								sc="s_"+str(s)
								if sc in scenes.keys():
									scenes[sc].update({obj_order[k]:{"rsconfig":c_start,"rgconfig":c_goal,"voxel":voxel, "one_hot":one_hot}, "env_no":env_no,"s_no":s_no, "obj_order":obj_order})
								else:
									scenes[sc]={obj_order[k]:{"rsconfig":c_start,"rgconfig":c_goal,"voxel":voxel, "one_hot":one_hot}, "env_no":env_no,"s_no":s_no, "obj_order":obj_order}
						s=s+1

	return scenes


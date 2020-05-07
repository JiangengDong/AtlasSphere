from openravepy import *
from numpy import *
import numpy as np
from str2num import *
from rodrigues import *
from TransformMatrix import *
from TSR import *
import time
import sys
import math
import copy
import pickle
import os

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.misc

import open3d as o3d

#####################

def setup_esc(orEnv,targObj,obj_names,e_no,s_no,esc_dict):
        env_no="env_"+str(e_no)
        sc_no="s_"+str(s_no)
        targets=esc_dict[env_no]["targets"]
        scene=esc_dict[env_no][sc_no]
        for i in range(0,2):
                orEnv.Add(targObj[i])
                T0_object=targets[obj_names[i]]["T0_w2"]
                targObj[i].SetTransform(array(T0_object[0:3][:,0:4]))

        for i in range(2,len(targObj)):
                orEnv.Add(targObj[i])
                T0_object=scene[obj_names[i]]["T0_w"]
                targObj[i].SetTransform(array(T0_object[0:3][:,0:4]))

def display_esc(e_no,s_no, orEnv,targobject,obj_names,esc_dict):
        for i in range(0,e_no):
                for j in range(0,s_no):
                        setup_esc(orEnv,targobject,obj_names,i,j,esc_dict)
                        env_no="env_"+str(i)
                        sc_no="s_"+str(j)
                        print("env_no:"+env_no+" s_no:"+sc_no)
                        print(esc_dict[env_no][sc_no]["obj_order"])
                        time.sleep(4)

'''def pointcloud_to_voxel(points, voxel_size=(24, 24, 24), padding_size=(32, 32, 32)):
    voxels = [voxelize(points[i], voxel_size, padding_size) for i in range(len(points))]
    # return size: BxV*V*V
    return np.array(voxels)

def voxelize(points, voxel_size=(24, 24, 24), padding_size=(32, 32, 32), resolution=0.05):
    """
    Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
    `padding_to_size`. The outside part is cut, rather than scaling the points.

    Args:
    `points`: pointcloud in 3D numpy.ndarray (shape: N * 3)
    `voxel_size`: the centerlized voxel size, default (24,24,24)
    `padding_to_size`: the size after zero-padding, default (32,32,32)
    `resolution`: the resolution of voxel, in meters

    Ret:
    `voxel`:32*32*32 voxel occupany grid
    `inside_box_points`:pointcloud inside voxel grid
    """
    # calculate resolution based on boundary
    if abs(resolution) < sys.float_info.epsilon:
        print('error input, resolution should not be zero')
        return None, None

    """
    here the point cloud is centerized, and each dimension uses a different resolution
    """
    resolution = [(points[:,i].max() - points[:,i].min()) / voxel_size[i] for i in range(3)]
    resolution = np.array(resolution)
    #resolution = np.max(res)
    # remove all non-numeric elements of the said array
    points = points[np.logical_not(np.isnan(points).any(axis=1))]

    # filter outside voxel_box by using passthrough filter
    # TODO Origin, better use centroid?
    origin = (np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2]))
    # set the nearest point as (0,0,0)
    points[:, 0] -= origin[0]
    points[:, 1] -= origin[1]
    points[:, 2] -= origin[2]
    # logical condition index
    x_logical = np.logical_and((points[:, 0] < voxel_size[0] * resolution[0]), (points[:, 0] >= 0))
    y_logical = np.logical_and((points[:, 1] < voxel_size[1] * resolution[1]), (points[:, 1] >= 0))
    z_logical = np.logical_and((points[:, 2] < voxel_size[2] * resolution[2]), (points[:, 2] >= 0))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    inside_box_points = points[xyz_logical]

    # init voxel grid with zero padding_to_size=(32*32*32) and set the occupany grid
    voxels = np.zeros(padding_size)
    # centerlize to padding box
    center_points = inside_box_points + (padding_size[0] - voxel_size[0]) * resolution / 2
    # TODO currently just use the binary hit grid
    x_idx = (center_points[:, 0] / resolution[0]).astype(int)
    y_idx = (center_points[:, 1] / resolution[1]).astype(int)
    z_idx = (center_points[:, 2] / resolution[2]).astype(int)
    voxels[x_idx, y_idx, z_idx] = OCCUPIED
    return voxels'''

def cam_image(kinect, camera):
	#camera.Configure(Sensor.ConfigureCommand.PowerOn)
	img = kinect.GetAttachedSensors()[1].GetData().imagedata
	return img
	


def voxelize_scene(kinect1, sensor1, kinect2, sensor2, vox_size, disp=False):


	sensor1.Configure(Sensor.ConfigureCommand.PowerOn)
	sensor2.Configure(Sensor.ConfigureCommand.PowerOn)

	time.sleep(2.5)
	data1 = kinect1.GetAttachedSensors()[0].GetData()
	data2 = kinect2.GetAttachedSensors()[0].GetData()
	sensor1_pos = data1.positions
	sensor1_pc = data1.ranges
	sensor2_pos = data2.positions
	sensor2_pc = data2.ranges

	# threshold both point clouds
	dist1 = linalg.norm(sensor1_pc, axis=1)
	dist2 = linalg.norm(sensor2_pc, axis=1)
	#sensor1_min_thresh = sensor1_pc[dist1>=0.1, :]
	#sensor2_min_thresh = sensor2_pc[dist1>=0.1, :]

	#sensor1_pc_thresh = sensor1_min_thresh[dist1<=5, :] + sensor1_pos
	#sensor2_pc_thresh = sensor2_min_thresh[dist2<=5, :] + sensor2_pos

	sensor1_pc_thresh = sensor1_pc[dist1<=5, :] + sensor1_pos
	sensor2_pc_thresh = sensor2_pc[dist2<=5, :] + sensor2_pos

	sensor1.Configure(Sensor.ConfigureCommand.PowerOff)
	sensor2.Configure(Sensor.ConfigureCommand.PowerOff)

	# merge and plot point cloud
	point_cloud = vstack((sensor1_pc_thresh,sensor2_pc_thresh))
	#print(np.shape(point_cloud))
	#voxels = pointcloud_to_voxel(point_cloud)


	# plot point cloud
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], marker = '.', s=0.01)
	if disp:
		plt.show()


	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(point_cloud)
	#o3d.visualization.draw_geometries([pcd])

	#downpcd=pcd.voxel_down_sample(voxel_size=vox_size)

	downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=vox_size)
	#o3d.visualization.draw_geometries([downpcd])

	#voxel_grid = pcd.create_surface_voxel_grid_from_point_cloud(voxel_size=vox_size)

	voxel_grid = o3d.geometry.create_surface_voxel_grid_from_point_cloud(downpcd, voxel_size=vox_size)
	#voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(downpcd, voxel_size=vox_size)
	voxels = voxel_grid.voxels
	coords = zeros((len(voxels),3))
	for i in range(0, len(voxels)):
    		coords[i,:] = voxels[i]#.grid_index

	coords = coords.astype(int)
	#print coords
	x_max = max(coords[:,0])
	y_max = max(coords[:,1])
	z_max = max(coords[:,2])
	#print "max in x: ", x_max, ", max in y: ", y_max, ", max in z: ", z_max
	#print "min in x: ", min(coords[:,0]), ", min in y: ", min(coords[:,1]), ", min in z: ", min(coords[:,2])
	ubound = max(x_max, y_max, z_max)
	#print ubound

	voxel_scene = zeros((ubound+1,ubound+1,ubound+1))
	for i in range(0, coords.shape[0]):
    		#print i, coords[i,:]
    		voxel_scene[coords[i,0],coords[i,1],coords[i,2]] = 1
	return voxel_scene
	



####################


urdf_path = "/home/austinchoe/catkin_ws/src/baxter_common/baxter_description/urdf/baxter_sym.urdf"
srdf_path = "/home/austinchoe/catkin_ws/src/baxter_common/baxter_description/urdf/baxter_new.srdf"

orEnv = Environment()
orEnv.SetViewer('qtcoin')

orEnv.Reset()
module = RaveCreateModule(orEnv, 'urdf')

orEnv.SetDebugLevel(DebugLevel.Info)
colchecker = RaveCreateCollisionChecker(orEnv,'ode')
orEnv.SetCollisionChecker(colchecker)

# list of objects
obj_names=["tray","recyclingbin","juice","fuze_bottle","coke_can","plasticmug","teakettle"]
targobject=[]
targobject.append(orEnv.ReadKinBodyXMLFile('../../ormodels/objects/household/tray.kinbody.xml'))
targobject.append(orEnv.ReadKinBodyXMLFile('../../ormodels/objects/household/recyclingbin.kinbody.xml'))
targobject.append(orEnv.ReadKinBodyXMLFile('../../ormodels/objects/household/juice_bottle.kinbody.xml'))
targobject.append(orEnv.ReadKinBodyXMLFile('../../ormodels/objects/household/fuze_bottle.kinbody.xml'))
targobject.append(orEnv.ReadKinBodyXMLFile('../../ormodels/objects/household/coke_can.kinbody.xml'))
targobject.append(orEnv.ReadKinBodyXMLFile('../../ormodels/objects/household/mug2.kinbody.xml'))
targobject.append(orEnv.ReadKinBodyXMLFile('../../ormodels/objects/household/teakettle.kinbody.xml'))


# add tables
tables=[]
table1 = RaveCreateKinBody(orEnv,'')
table1.SetName('table1')
table1.InitFromBoxes(numpy.array([[1.0, 0.4509, 0.5256, 0.2794, 0.9017, 0.5256]]),True)
orEnv.Add(table1,True)
table2 = RaveCreateKinBody(orEnv,'')
table2.SetName('table2')
table2.InitFromBoxes(numpy.array([[0.3777, -0.7303, 0.5256, 0.9017, 0.2794, 0.5256]]),True)
orEnv.Add(table2,True)
tables.append(table1)
tables.append(table2)

# add kinect
kinect1 = orEnv.ReadRobotURI('kinectv2.robot.xml')
kinect1.SetName('kinect1')
orEnv.Add(kinect1,True)
kinect2 = orEnv.ReadRobotURI('kinectv2.robot.xml')
kinect2.SetName('kinect2')
#orEnv.Add(kinect2,True)

kinect_rot1 = [[0,0,1],[0,1,0],[-1,0,0]]
#kinect_rot2 = [[0,1,0],[0,0,-1],[-1,0,0]]
#T0_kinect1 = MakeTransform(mat(kinect_rot1),mat([1.0, 0.4509, 2.5256]).T)
T0_kinect1 = MakeTransform(mat(kinect_rot1),mat([0.4, 0.2509, 2.5256]).T)
#T0_kinect2 = MakeTransform(mat(kinect_rot2),mat([0.3777, -0.7303, 2.5256]).T)
kinect1.SetTransform(array(T0_kinect1[0:3][:,0:4]))
#kinect2.SetTransform(array(T0_kinect2[0:3][:,0:4]))

sensor1 = orEnv.GetSensor('kinect1:kinect_lidar')
#sensor2 = orEnv.GetSensor('kinect2:kinect_lidar')
camera1 = orEnv.GetSensor('kinect1:kinect_rgb')
#camera2 = orEnv.GetSensor('kinect2:kinect_rgb')
camera1.Configure(Sensor.ConfigureCommand.PowerOn)
#camera2.Configure(Sensor.ConfigureCommand.PowerOn)

### when loading files
#esc_dict = pickle.load( open( "esp_dict0_19.p", "rb" ) )
esc_dict = pickle.load( open( "esc_dict20_120.p", "rb" ) )
## iterate through all environments and scenes
#i=2 # env_no
#j=4     # sc_no
time.sleep(5.0)
esc_img={} 
for i in range(0,20):
	sc_img={} 
	env_no="env_"+str(i) 
	for j in range(60,120):
		s_no="s_"+str(j)
		if s_no in esc_dict[env_no]:
			print("env_no:"+env_no+" s_no:"+s_no)
			obj_order=esc_dict[env_no][s_no]["obj_order"]
			setup_esc(orEnv,targobject,obj_names,i,j,esc_dict)
			time.sleep(0.5)
			img_full = cam_image(kinect1, camera1)

			#scipy.misc.imsave("cam3/cam3_"+env_no+"_"+s_no+".png",img1)
			for k in range(0,len(obj_order)):
				idx=obj_names.index(obj_order[k])
				if obj_order[k] =="teakettle" or obj_order[k] == "plasticmug":
					T0_object=esc_dict[env_no]["targets"][obj_names[idx]]["T0_w2"]
					targobject[idx].SetTransform(array(T0_object[0:3][:,0:4]))
					#input("enter2")
					time.sleep(1.0)
				else:
					print(targobject[idx])
					print("++++++++++++++++++remove "+ obj_order[k])
					orEnv.Remove(targobject[idx])
					#input("enter3")
					time.sleep(1.0)

				print("saving camera images")
				img = cam_image(kinect1, camera1)
				#print(np.shape(img))
				#scipy.misc.imsave("cam3/cam3_"+env_no+"_"+s_no+"_"+str(obj_order[k])+".png",img1)			
				if s_no in sc_img.keys(): # hard-coded to env2_sc4 for testing
					sc_img[s_no].update({obj_order[k]:img})
				else: # hard-coded to env2_sc4 for testing
					sc_img[s_no]={obj_order[k]:img,"full":img_full}
					#sc_img[s_no]={"full":img_full}
			

				#scipy.misc.imsave("cam2/cam2_"+env_no+"_"+s_no+"_"+str(obj_order)+".png",img2)
	
	if env_no in esc_img.keys():
		esc_img[env_no].update(sc_img)
	else:
		esc_img[env_no]=sc_img
	pickle.dump( esc_img, open( "esimg_dict0_19_120.pkl", "wb" ) )
print "Press return to exit"
sys.stdin.readline()


#! /usr/bin/env python
# coding=utf-8

import os
import argparse
import numpy as np
from easydict import EasyDict

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

import tf
import message_filters
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from yolov5.utils.torch_utils import select_device
from yolov5.detect import yolov5_inference
from yolov5.models.experimental import attempt_load
from OpenPCDet.demo_rod import DemoDataset, openpcdet_inference
from OpenPCDet.pcdet.models import build_network, load_data_to_gpu
from OpenPCDet.pcdet.config import cfg, cfg_from_yaml_file
from OpenPCDet.pcdet.utils import common_utils

import time
import yaml
import numba

import sys
### python3 opencv module might contradict with python2.7, this line might help
### because generally 'source /opt/ros/kinetic/setup.bash' will add python2.7 path into PYTHONPATH
### then ROS's opencv is prior to be found
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
### but personally recommand to add anaconda env in front of others in PYTHONPATH
# eg: export PYTHONPATH=/xxx/anaconda3/envs/xxx/lib/pythonx.x/site-packages:$PYTHONPATH
import cv2 as cv


# get the params for yolov5 model
def yolov5_parse_cfg(cfg_file):
	with open(cfg_file, 'r') as f:
		try:
			config = yaml.safe_load(f, Loader=yaml.FullLoader)
		except:
			config = yaml.safe_load(f)
	config = EasyDict(config)
	return config


# get the params for openpcdet model
def openpcdet_parse_cfg(cfg_file):
	with open(cfg_file, 'r') as f:
		try:
			config = yaml.safe_load(f, Loader=yaml.FullLoader)
		except:
			config = yaml.safe_load(f)

	config = EasyDict(config)
	cfg_from_yaml_file(config.cfg_file, cfg)

	return config, cfg


# pad the fourth row for transformation matrix
def cart_to_hom(pts):
	"""
	:param pts: (N, 3 or 2)
	:return pts_hom: (N, 4 or 3)
	"""
	pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
	return pts_hom





### GLOBAL VARIABLES
# ROS callback function is asynchronous operation
# tried put these variables in main function, and refer with 'global' in callback function, but didn't work
# so define these variables here

# params setting
yolov5_cfg_file = '/home/fengchen/projects/ros-yolov5-pointpillar/src/rospy_sensorfusion/config/Yolov5.yaml'
openpcdet_cfg_file = '/home/fengchen/projects/ros-yolov5-pointpillar/src/rospy_sensorfusion/config/OpenPCDet.yaml'
yolov5_cfg = yolov5_parse_cfg(yolov5_cfg_file)
openpcdet_cfg, cfg = openpcdet_parse_cfg(openpcdet_cfg_file)

## produce EasyDict used to initialize openpcdet_model
# originally should use a dataset object, here use a EasyDict object to simplify
# (EasyDict can hierachically get attributes with '.')
dataset_cfg = EasyDict()
point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
voxel_size = cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE
grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
grid_size = np.round(grid_size).astype(np.int64)
dataset_cfg['class_names'] = cfg.CLASS_NAMES
dataset_cfg['point_feature_encoder'] = {'num_point_features' : cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES}
dataset_cfg['point_cloud_range'] = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
dataset_cfg['voxel_size'] = cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE
dataset_cfg['grid_size'] = grid_size
dataset_cfg['depth_downsample_factor'] = None

## Load model
# yolov5 model
weights = yolov5_cfg.weights
device = select_device()
yolov5_model = attempt_load(weights, map_location=device)  # load FP32 model
yolov5_model.eval()
# openpcdet model - pointpillar
logger = common_utils.create_logger()
# openpcdet_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
openpcdet_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset_cfg)
openpcdet_model.load_params_from_file(filename=openpcdet_cfg.ckpt, logger=logger, to_cpu=True)
openpcdet_model.cuda()
openpcdet_model.eval()

## define publisher
boundingboxarr_publisher = rospy.Publisher('/sensor_fusion/detection_bbox', BoundingBoxArray, queue_size=5)
detection_img_publisher = rospy.Publisher('/sensor_fusion/detection_img', Image, queue_size=5)
sync_pcl_publisher = rospy.Publisher('/sensor_fusion/sync_pcl', PointCloud2, queue_size=5)
# sync_img_publisher = rospy.Publisher('/sensor_fusion/sync_img', Image, queue_size=5)


# numba accelerate the semantic information adding process
# might realize in matrix operation, could modify later
@numba.jit
def sem_process(pts_image, sem_img, pad, sem_car, sem_pedestrian, sem_bus):
	for i, pts in enumerate(pts_image):
		y, x = pts[:-1]
		if x >= 0 and x < sem_img.shape[0] and y >= 0 and y < sem_img.shape[1]:
			if sem_img[x, y] == sem_pedestrian:
				pad[i][2] = 1
			elif sem_img[x, y] == sem_car or sem_img[x, y] == sem_bus: 
				pad[i][1] = 1
		else:
			pad[i][0] = 1
	return pad


# the main callback function of whole procedure
def process(pcl_msg, img_msg):

	# mention global variables
	global yolov5_cfg, openpcdet_cfg, dfg, dataset_cfg, yolov5_model, logger, openpcdet_model
	global boundingboxarr_publisher, detection_img_publisher, sync_pcl_publisher

	############################ image detection ############################

	start_time = time.time()

	### MSG to data process
	# pcl_msg to pcl array
	pcl = pc2.read_points(pcl_msg, skip_nans=True, field_names=('x', 'y', 'z', 'intensity'))
	pcl_list = []
	for p in pcl:
		pcl_list.append([p[0], p[1], p[2], p[3]])
	pcl_arr = np.array(pcl_list)  # [n, 4]
	# img_msg to img array
	bridge = CvBridge()
	img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')

	### yolov5 model inference
	sem_img, detect_img = yolov5_inference(img, yolov5_model, **vars(yolov5_cfg))

	### img result publish
	detect_img_msg = bridge.cv2_to_imgmsg(detect_img, 'bgr8')
	detection_img_publisher.publish(detect_img_msg)

	yolov5_time = time.time()

	######################### produce semantic pcl ##########################

	### intensity normalization
	# road side ROS pcl data's intensity range from (0, 255)
	pcl_arr[:, -1] /= 255.0

	### projection points to image
	# 9th rod transformation matrix, get from ctcc_trainee_xc-sd project
	# some can be found in params yaml file, some can get from console result
	TRANS_VELO_TO_CAM = np.array(
		[[-0.017847, -0.999472, -0.027126, 0.0657052],
		[-0.111906, 0.0289574, -0.993296, 0.40967],
		[0.993558, -0.0146925, -0.112363, -0.264861],
		[0, 0, 0, 1]])
	TRANS_RECTCAM_TO_IMAGE = np.array(
		[[2760.16, 0, 959.559, 0],
		[0, 2755.92, 591.173, 0],
		[0, 0, 1, 0]])
	TRANS_CAM_TO_RECTCAM = np.array(
		[[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]])

	# select points which x > 0, thus they are in image direction
	pcl_arr = pcl_arr[pcl_arr[:, 0] > 0]

	# use xyz data part of pcl 
	pcl_xyz = pcl_arr[:, :-1]  # [n, 3]

	# the matrix multiply order of ctcc road side and openpcdet are inverse
	# road side: T_matrix * [4, n]
	# openpcdet: [n, 4] * T_matrix.T
	# but the T_matrix are the same in the begining, no need to transpose
	pts_hom = cart_to_hom(pcl_xyz)  # [n, 3]

	# get points in image coords
	pts_image = TRANS_RECTCAM_TO_IMAGE @ TRANS_CAM_TO_RECTCAM @ TRANS_VELO_TO_CAM @ pts_hom.T
	# pts_image = np.matmul(TRANS_RECTCAM_TO_IMAGE, np.matmul(TRANS_CAM_TO_RECTCAM, np.matmul(TRANS_VELO_TO_CAM, pts_hom.T)))
	pts_image = pts_image.T
	pts_image[:, 0] /= pts_image[:, -1]
	pts_image[:, 1] /= pts_image[:, -1]  # [n, 3]

	# sem info of class
	# [::-1] is for bgr to rgb, but deside to use sum result, no need here
	sem_car = np.sum([0, 0, 142][::-1])
	sem_pedestrian = np.sum([220, 20, 60][::-1])
	sem_bus = np.sum([0, 60, 100][::-1])

	### project point in image
	sem_img = sem_img.sum(axis=2)
	pts_image = pts_image.astype(np.int)

	# # filter points outside the image
	# but this operation will remove some obj around the boundary of the scene
	# so currently decide not to filter these points
	# x_cond = (0 <= pts_image[:, 1]) & (pts_image[:, 1] < sem_img.shape[0])
	# y_cond = (0 <= pts_image[:, 0]) & (pts_image[:, 0] < sem_img.shape[1])
	# cond = x_cond & y_cond
	# pcl_arr = pcl_arr[cond]
	# pts_image = pts_image[cond]

	# addition semantic feature arr
	pad = np.zeros((pcl_arr.shape[0], 4))

	### get semantic information for each point
	pad = sem_process(pts_image, sem_img, pad, sem_car, sem_pedestrian, sem_bus)

	pcl_sem_arr = np.concatenate((pcl_arr, pad), axis=1)

	project_time = time.time()

	######################### semantic pcl data detection ##########################

	### adjust height of road side testing pcl data
	# pcl height += 4 when testing
	pcl_sem_arr[:, 2] += 4

	# turn the pcl_sem_arr into kitti demo_rod.py DemoDataset to inference
	# refer to the demo_rod.py or demo.py, this object is in need in inference
	pcl_dataset = DemoDataset(
		dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
		root_path=None, ext=openpcdet_cfg.ext, logger=logger, points=pcl_sem_arr
	)

	### openpcdet model inference
	# ['pred_boxes'], ['pred_scores'], ['pred_labels'] -- (N, 7), N, N 
	# label is class_id
	# boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
	pred_boxes, pred_scores, pred_labels = openpcdet_inference(openpcdet_model, pcl_dataset)

	pointpillar_time = time.time()

	### ROS jsk_recognition_msg to publish 3D detection results
	boundingboxarr = BoundingBoxArray()
	boundingboxarr.header = pcl_msg.header
	for i in range(pred_boxes.shape[0]):
		boundingbox = BoundingBox()

		pred_box = pred_boxes[i]
		pred_score = pred_scores[i]
		pred_label = pred_labels[i]

		boundingbox.header = pcl_msg.header
		boundingbox.pose.position.x = pred_box[0]
		boundingbox.pose.position.y = pred_box[1]
		boundingbox.pose.position.z = pred_box[2]
		# heading is a single value result
		# tf module is need to transform eular angle into quaternion value: x y z w
		quaternion = tf.transformations.quaternion_from_euler(0, 0, pred_box[6])
		boundingbox.pose.orientation.x = quaternion[0]
		boundingbox.pose.orientation.y = quaternion[1]
		boundingbox.pose.orientation.z = quaternion[2]
		boundingbox.pose.orientation.w = quaternion[3]
		boundingbox.dimensions.x = pred_box[3]
		boundingbox.dimensions.y = pred_box[4]
		boundingbox.dimensions.z = pred_box[5]
		boundingbox.value = pred_score
		boundingbox.label = pred_label

		boundingboxarr.boxes.append(boundingbox)

	boundingboxarr_publisher.publish(boundingboxarr)

	### synchronized msg for visualization
	## sync pcl msg publish
	sync_pcl_publisher.publish(pcl_msg)
	## sync img msg might not in need, cause img plotted boxes is published above
	# sync_img_publisher.publish(img_msg)

	end_time = time.time()

	print('yolov5: ', yolov5_time - start_time, ' project: ', project_time - yolov5_time, 
		' pointpillar: ', pointpillar_time - project_time, ' total: ', end_time - start_time)


if __name__ == '__main__':
	
	rospy.init_node('rospy_sensorfusion', anonymous=True)
	# subscriber of 9th rod img and pcl data
	pcl_sub = message_filters.Subscriber('/rslidar_points', PointCloud2)
	img_sub = message_filters.Subscriber('/ue09_pylon02/image_raw', Image)

	# timestamp approximate synchronization object
	sync = message_filters.ApproximateTimeSynchronizer([pcl_sub, img_sub], 10, 0.1, allow_headerless=True)
	sync.registerCallback(process)

	### the information a openpcdet dataset must contain when initialize a model
	### I use a EasyDict object to substitute
	# model_info_dict = {
	# 	'module_list': [],
	# 	'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
	# 	'num_point_features': self.dataset.point_feature_encoder.num_point_features,
	# 	'grid_size': self.dataset.grid_size,
	# 	'point_cloud_range': self.dataset.point_cloud_range,
	# 	'voxel_size': self.dataset.voxel_size,
	# 	'depth_downsample_factor': self.dataset.depth_downsample_factor
	# }

	try:
		rospy.spin()
	except:
		print("over!")
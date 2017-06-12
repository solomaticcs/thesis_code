import ntpath
import os
import sys
import cv2
import myfunc
import re
import numpy as np
import json
import cPickle as pickle

def pickle_keypoints(keypoints, descriptors):
	i = 0
	temp_array = []
	for point in keypoints:
		temp = (point.pt, point.size, point.angle, point.response, point.octave,
		point.class_id, descriptors[i])     
		++i
		temp_array.append(temp)
	return temp_array

def unpickle_keypoints(array):
	keypoints = []
	descriptors = []
	for point in array:
		temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
		temp_descriptor = point[6]
		keypoints.append(temp_feature)
		descriptors.append(temp_descriptor)
	return keypoints, np.array(descriptors)

def create_des(filepath,outputfolderpath):
	print outputfolderpath
	# get filename
	filename = myfunc.path_leaf(filepath)
	# get basename
	basename= myfunc.path_name(filename)

	# get wine label
	winelabel = myfunc.find_wine_label_v2(filepath,False)
	# get wine height, width
	height, width = winelabel.shape[:2]
	if width < 250 or height < 250:
		# use origin image
		winelabel = cv2.imread(filepath)
	# convert image to gray
	winelabelgray = cv2.cvtColor(winelabel,cv2.COLOR_BGR2GRAY)

	# get orb descriptors
	orb = cv2.ORB_create()
	orb_kp, orb_des = orb.detectAndCompute(winelabelgray,None)
	pickle.dump(pickle_keypoints(orb_kp,orb_des), open(outputfolderpath+"/"+basename+".orb", "wb"))

	# get sift descriptors
	sift = cv2.xfeatures2d.SIFT_create()
	sift_kp, sift_des = sift.detectAndCompute(winelabelgray,None)
	pickle.dump(pickle_keypoints(sift_kp,sift_des), open(outputfolderpath+"/"+basename+".sift", "wb"))

	# get surf descriptors
	surf = cv2.xfeatures2d.SURF_create()
	surf_kp, surf_des = surf.detectAndCompute(winelabelgray,None)
	pickle.dump(pickle_keypoints(surf_kp,surf_des), open(outputfolderpath+"/"+basename+".surf", "wb"))

if len(sys.argv)>1:
	# get image folder
	imgfolderpath=sys.argv[1]
	# get output folder
	outputfolderpath=sys.argv[2]
	# create directory
	if not os.path.exists(outputfolderpath):
		os.makedirs(outputfolderpath)
	# get image folder files name
	files = os.listdir(imgfolderpath)
	# order files
	ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
	for filename in ordered_files:
		filepath = imgfolderpath + "/" + filename
		print filepath
		create_des(filepath,outputfolderpath)

else:
	print "error"

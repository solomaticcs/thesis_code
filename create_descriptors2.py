import ntpath
import os
import sys
import cv2
import myfunc
import re
import numpy as np

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
	kp, orb_des = orb.detectAndCompute(winelabelgray,None)
	orb_des_height, orb_des_width = orb_des.shape[:2]
	print str(orb_des_height) + "," + str(orb_des_width)
	orb_des_arr=[orb_des_height,orb_des_width]
	for i in range(0,orb_des_height,1):
		for j in range(0,orb_des_width,1):
			orb_des_arr.append(orb_des[i][j])
	orb_file = open(outputfolderpath+"/"+basename+"_orb.deslist", 'w')
	orb_file.write(str(orb_des_arr))
	orb_file.close()

	# get sift descriptors
	sift = cv2.xfeatures2d.SIFT_create()
	kp, sift_des = sift.detectAndCompute(winelabelgray,None)
	sift_des_height, sift_des_width = sift_des.shape[:2]
	print str(sift_des_height) + "," + str(sift_des_width)
	sift_des_arr=[sift_des_height,sift_des_width]
	for i in range(0,sift_des_height,1):
		for j in range(0,sift_des_width,1):
			sift_des_arr.append(sift_des[i][j])
	sift_file = open(outputfolderpath+"/"+basename+"_sift.deslist", 'w')
	sift_file.write(str(sift_des_arr))
	sift_file.close()

	# get surf descriptors
	surf = cv2.xfeatures2d.SURF_create()
	kp, surf_des = surf.detectAndCompute(winelabelgray,None)
	surf_des_height, surf_des_width = surf_des.shape[:2]
	print str(surf_des_height) + "," + str(surf_des_width)
	surf_des_arr=[surf_des_height,surf_des_width]
	for i in range(0,surf_des_height,1):
		for j in range(0,surf_des_width,1):
			surf_des_arr.append(surf_des[i][j])
	surf_file = open(outputfolderpath+"/"+basename+"_surf.deslist", 'w')
	surf_file.write(str(surf_des_arr))
	surf_file.close()


if len(sys.argv)>1:
	# get image folder
	imgfolderpath=sys.argv[1]
	# get output folder
	outputfolderpath=sys.argv[2]
	# create directory
	if not os.path.exists(outputfolderpath):
		os.makedirs(outputfolderpath)

	files = os.listdir(imgfolderpath)
	ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
	for filename in ordered_files:
		filepath = imgfolderpath + "/" + filename
		print filepath
		create_des(filepath,outputfolderpath)

else:
	print "error"

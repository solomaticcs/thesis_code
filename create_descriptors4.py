import ntpath
import os
import sys
import cv2
import myfunc
import re
import numpy as np
import json

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
	# if width < 250 or height < 250:
	# 	# use origin image
	# 	winelabel = cv2.imread(filepath)
	# convert image to gray
	if height > 0 and width > 0:
		print "height > 0 and width > 0"
		winelabelgray = cv2.cvtColor(winelabel,cv2.COLOR_BGR2GRAY)

		# get orb descriptors
		orb = cv2.ORB_create()
		kp, orb_des = orb.detectAndCompute(winelabelgray,None)
		if orb_des != None:
			np.save(outputfolderpath+"/"+basename+"_orb",orb_des)
		else:
			np.save(outputfolderpath+"/"+basename+"_orb",np.zeros(0))
			
		# get sift descriptors
		sift = cv2.xfeatures2d.SIFT_create()
		kp, sift_des = sift.detectAndCompute(winelabelgray,None)
		if sift_des != None:
			np.save(outputfolderpath+"/"+basename+"_sift",sift_des)
		else:
			np.save(outputfolderpath+"/"+basename+"_sift",np.zeros(0))

		# get surf descriptors
		surf = cv2.xfeatures2d.SURF_create()
		kp, surf_des = surf.detectAndCompute(winelabelgray,None)
		if surf_des != None:
			np.save(outputfolderpath+"/"+basename+"_surf",surf_des)
		else:
			np.save(outputfolderpath+"/"+basename+"_surf",np.zeros(0))

	else:
		print "give zeros"
		np.save(outputfolderpath+"/"+basename+"_orb",np.zeros(0))
		np.save(outputfolderpath+"/"+basename+"_sift",np.zeros(0))
		np.save(outputfolderpath+"/"+basename+"_surf",np.zeros(0))

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

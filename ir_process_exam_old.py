import ntpath
import os
import sys
from PIL import Image 
import pytesseract
import cv2
import myfunc
import re
import time

def ir(filepath,imgfolderpath):

	# get filename
	filename = myfunc.path_leaf(filepath)

	# create directory
	wine_directory="./output/" + filename;
	if not os.path.exists(wine_directory):
		os.makedirs(wine_directory)

	# get wine label
	winelabel1 = myfunc.find_wine_label_v2(filepath,False)
	# convert image to gray
	winelabelgray1 = cv2.cvtColor(winelabel1,cv2.COLOR_BGR2GRAY)


	_file = open(wine_directory+"/"+filename+"_match_image.txt", "w")
	

	""" image Recognition """

	# init
	orb = cv2.ORB_create()
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# get descriptors
	kp1 = orb.detect(winelabelgray1,None)
	kp1, des1 = orb.compute(winelabelgray1,kp1)

	# match images
	match_filename = "none"
	max_match_point=0
	DIST_LIMIT = 45
	print "START ORB"
	orb_start_time = time.time()
	files = os.listdir(imgfolderpath)
	ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
	for dirname in ordered_files:
		if dirname == filename:
			continue
		print "filename:" + filename + " dirname:" + dirname
		winelabel2 = myfunc.find_wine_label(imgfolderpath+"/"+dirname,False)
		winelabelgray2 = cv2.cvtColor(winelabel2,cv2.COLOR_BGR2GRAY)
		kp2 = orb.detect(winelabelgray2,None)
		kp2, des2 = orb.compute(winelabelgray2,kp2)
		
		if len(kp1) != 0 and des1 != None and len(kp2) != 0 and des2 != None:
			# get two image's match
			good = []
			matches = bf.match(des1,des2)
			for m in matches:
				if m.distance < DIST_LIMIT:
					good.append(m)

			print "ORB MATCH POINTS:" + str(len(good))

			if max_match_point < len(good):
				max_match_point = len(good)
				match_filename = dirname
	
	output_str = filename + "| ORB | match_filename:" + match_filename + " max_match_point:" +str(max_match_point)
	print output_str
	orb_end_time = time.time()
 	orb_time =  orb_end_time-orb_start_time
 	output_str2 = "orb time:" + str(orb_time) + " seconds"
 	print output_str2
	_file.write(output_str+"\n")
	_file.write(output_str2+"\n")

	height1,width1 = winelabel1.shape[:2]

	if height1>0 and width1>0:
		# init
		sift = cv2.xfeatures2d.SIFT_create()
		bf = cv2.BFMatcher()

		# get descriptors
		kp1, des1 = sift.detectAndCompute(winelabelgray1,None)

		# match images
		match_filename = "none"
		max_match_point=0
		print "START SIFT"
		sift_start_time = time.time()
		files = os.listdir(imgfolderpath)
		ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
		for dirname in ordered_files:
			if dirname == filename:
				continue
			print "filename:" + filename + " dirname:" + dirname
			winelabel2 = myfunc.find_wine_label(imgfolderpath+"/"+dirname,False)
			height2,width2 = winelabel2.shape[:2]
			if width2 > 0 and height2 > 0:
				winelabelgray2 = cv2.cvtColor(winelabel2,cv2.COLOR_BGR2GRAY)
				kp1, des2 = sift.detectAndCompute(winelabelgray2,None)

				if len(kp1) != 0 and des1 != None and len(kp2) != 0 and des2 != None:
					matches = bf.knnMatch(des1,des2, k=2)
					good = []
					for m_n in matches:
						if len(m_n) != 2:
							continue
						(m,n) = m_n
						if m.distance < 0.75*n.distance:
							good.append(m)

					print "SIFT MATCH POINTS:" + str(len(good))

					if max_match_point < len(good):
						max_match_point = len(good)
						match_filename = dirname

		output_str = filename + " | SIFT |match_filename:" + match_filename + " max_match_point:" +str(max_match_point)
		print output_str
		sift_end_time = time.time()
		sift_time = sift_end_time-sift_start_time
		output_str2 = "sift time:" + str(sift_time) + " seconds"
		print output_str2
		_file.write(output_str+"\n")
		_file.write(output_str2+"\n")

		#init
		surf = cv2.xfeatures2d.SURF_create()
		bf = cv2.BFMatcher()

		# get descriptors
		kp1, des1 = surf.detectAndCompute(winelabelgray1,None)

		# match images
		match_filename = "none"
		max_match_point=0
		print "START SURF"
		surf_start_time = time.time()
		files = os.listdir(imgfolderpath)
		ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
		for dirname in ordered_files:
			if dirname == filename:
				continue
			print "filename:" + filename + " dirname:" + dirname
			winelabel2 = myfunc.find_wine_label(imgfolderpath+"/"+dirname,False)
			height2,width2 = winelabel2.shape[:2]
			if width2 > 0 and height2 > 0:
				winelabelgray2 = cv2.cvtColor(winelabel2,cv2.COLOR_BGR2GRAY)
				kp2, des2 = surf.detectAndCompute(winelabelgray2,None)

				if len(kp1) != 0 and des1 != None and len(kp2) != 0 and des2 != None:
					matches = bf.knnMatch(des1,des2, k=2)
					good = []
					for m_n in matches:
						if len(m_n) != 2:
							continue
						(m,n) = m_n
						if m.distance < 0.75*n.distance:
							good.append(m)

					print "SURF MATCH POINTS:" + str(len(good))

					if max_match_point < len(good):
						max_match_point = len(good)
						match_filename = dirname

		output_str = filename + " | SURF | match_filename:" + match_filename + " max_match_point:" +str(max_match_point)
		print output_str
		surf_end_time = time.time()
		surf_time = surf_end_time-surf_start_time
		output_str2 = "surf time:" + str(surf_time) + " seconds"
		print output_str2
		_file.write(output_str+"\n")
		_file.write(output_str2+"\n")
	else:
		output_str = filename + "SIFT |match_filename:none max_match_point:0"
		_file.write(output_str+"\n")
		_file.write("sift time: 0 seconds")
		output_str = filename + "SURF | match_filename:none max_match_point:0"
		_file.write(output_str+"\n")
		_file.write("surf time: 0 seconds")
	_file.close()

	print "====="+filename+" END====="

if len(sys.argv)>2:
	testimgfolderpath=sys.argv[1]
	print testimgfolderpath
	trainimgfolderpath=sys.argv[2]
	print trainimgfolderpath

	files = os.listdir(testimgfolderpath)
	ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
	for filename in ordered_files:
		filepath = testimgfolderpath+"/"+filename
		print filepath
		ir(filepath, trainimgfolderpath)

else:
	print "arg1: testimgfolderpath arg2: trainimgfolderpath"

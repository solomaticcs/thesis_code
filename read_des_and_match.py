import ntpath
import os
import sys
import cv2
import myfunc
import re
import numpy as np

# this python file is for test
# read des list and transform to nparray for match

def main():
	if len(sys.argv)>1:
		# # get des list folderpath
		# deslist_folderpath=sys.argv[1]
		# # get output folderpath
		# outputfolderpath=sys.argv[2]
		# # create directory
		# if not os.path.exists(outputfolderpath):
		# 	os.makedirs(outputfolderpath)
		
		# files = os.listdir(deslist_folderpath)
		# ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
		# for filename in ordered_files:
		# 	filepath = deslist_folderpath + "/" + filename
		# 	print filepath
		# 	read_des(filepath)

		global orb,sift,surf,bf1,bf2,orb_des_dist,sift_des_dist,surf_des_dist

		testdeslist_filepath=sys.argv[1]
		traindes_filepath=sys.argv[2]

		testfilename = myfunc.path_leaf(testdeslist_filepath)
		testname = myfunc.path_name(testfilename)
		testalg = testname.split("_")[1]
		
		trainfilename = myfunc.path_leaf(traindes_filepath)
		trainname = myfunc.path_name(trainfilename)
		trainalg = trainname.split("_")[1]

		test_des=myfunc.read_des(testdeslist_filepath,testalg)
		train_des=np.load(traindes_filepath)

		# """ init """
		orb = cv2.ORB_create()
		sift = cv2.xfeatures2d.SIFT_create()
		surf = cv2.xfeatures2d.SURF_create()
		bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		bf2 = cv2.BFMatcher()

		if testalg == "orb":
			""" Match ORB Features """
			DIST_LIMIT=45

			match_wineid = "none"
			max_match_point=0


			good = []
			matches = bf1.match(test_des,train_des)
			for m in matches:
				if m.distance < DIST_LIMIT:
					good.append(m)

			print len(good)
		elif testalg == "sift" or testalg == "surf":
			""" Match SURF or SIFT Features """
			matches = bf2.knnMatch(test_des,train_des, k=2)
			good = []
			for m_n in matches:
				if len(m_n) != 2:
					continue
				(m,n) = m_n
				if m.distance < 0.75*n.distance:
					good.append(m)

			print len(good)

	else:
		print "error"

main()
import ntpath
import os
import sys
import Image 
import pytesseract
import cv2
import myfunc

if len(sys.argv)>2:
	filepath1=sys.argv[1]
	print filepath1
	filename1=myfunc.path_leaf(filepath1)
	print filename1
	filepath2=sys.argv[2]
	print filepath2
	filename2=myfunc.path_leaf(filepath2)
	print filename2

	# get wine label
	winelabel1 = myfunc.find_wine_label(filepath1,False)
	winelabel2 = myfunc.find_wine_label(filepath2,False)
	# convert image to gray
	winelabelgray1 = cv2.cvtColor(winelabel1,cv2.COLOR_BGR2GRAY)
	winelabelgray2 = cv2.cvtColor(winelabel2,cv2.COLOR_BGR2GRAY)
	print winelabel1.shape
	print winelabel2.shape

	""" image Recognition """
	# init
	orb = cv2.ORB_create()
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# get descriptors
	kp1 = orb.detect(winelabelgray1,None)
	kp1, des1 = orb.compute(winelabelgray1,kp1)
	kp2 = orb.detect(winelabelgray2,None)
	kp2, des2 = orb.compute(winelabelgray2,kp1)

	# match images
	max_match_point=0
	DIST_LIMIT = 45
	
	# get two image's match
	good = []
	matches = bf.match(des1,des2)
	for m in matches:
		if m.distance < DIST_LIMIT:
			good.append(m)

	print "ORB Match Points:" + str(len(good))

	height1,width1 = winelabel1.shape[:2]
	height2,width2 = winelabel2.shape[:2]

	if height1>0 and width1>0 and height2>0 and width2>0:
		# init
		sift = cv2.xfeatures2d.SIFT_create()
		bf = cv2.BFMatcher()

		# get descriptors	
		kp1, des1 = sift.detectAndCompute(winelabelgray1,None)
		kp2, des2 = sift.detectAndCompute(winelabelgray2,None)

		# match images
		max_match_point=0
		DIST_LIMIT = 45
		
		if len(kp1) != 0 and des1 != None and len(kp2) != 0 and des2 != None:
			matches = bf.knnMatch(des1,des2, k=2)
			good = []
			for m_n in matches:
				if len(m_n) != 2:
					continue
				(m,n) = m_n
				if m.distance < 0.75*n.distance:
					good.append(m)

			print "SIFT Match Points:" +str(len(good))

		#init
		surf = cv2.xfeatures2d.SURF_create()
		bf = cv2.BFMatcher()

		# get descriptors
		kp1, des1 = surf.detectAndCompute(winelabelgray1,None)
		kp2, des2 = surf.detectAndCompute(winelabelgray2,None)

		# match images
		max_match_point=0
		DIST_LIMIT = 45
		if len(kp1) != 0 and des1 != None and len(kp2) != 0 and des2 != None:
			matches = bf.knnMatch(des1,des2, k=2)
			good = []
			for m_n in matches:
				if len(m_n) != 2:
					continue
				(m,n) = m_n
				if m.distance < 0.75*n.distance:
					good.append(m)

			print "SURF Match Points:" +str(len(good))

	print "=====END====="
else:
	print "arg1: filepath1, arg2: filepath2"

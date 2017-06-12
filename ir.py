import ntpath
import os
import sys
import Image 
import pytesseract
import cv2
import myfunc

if len(sys.argv)>3:
	filepath=sys.argv[1]
	print filepath
	filename=myfunc.path_leaf(filepath)
	print filename
	imgfolderpath=sys.argv[2]
	print imgfolderpath
	mode = sys.argv[3]
	print mode

	# create directory
	wine_directory="./output/" + filename;
	if not os.path.exists(wine_directory):
		os.makedirs(wine_directory)

	# get wine label
	winelabel = myfunc.find_wine_label(filepath,True)
	# convert image to gray
	winelabelgray = cv2.cvtColor(winelabel,cv2.COLOR_BGR2GRAY)

	_file = open(wine_directory+"/match_image.txt", "w")

	""" image Recognition """
	if mode == "orb":
		# init
		orb = cv2.ORB_create()
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# get descriptors
		kp = orb.detect(winelabelgray,None)
		kp, des = orb.compute(winelabelgray,kp)

		# match images
		max_match_point=0
		DIST_LIMIT = 45
		dirs = os.listdir(imgfolderpath)
		for dirname in dirs:
			print dirname
			dirimg = myfunc.find_wine_label(imgfolderpath+"/"+dirname,False)
			# print dirimg.shape
			dirgray = cv2.cvtColor(dirimg,cv2.COLOR_BGR2GRAY)
			dirkp = orb.detect(dirgray,None)
			dirkp, dirdes = orb.compute(dirgray,dirkp)
			
			if len(dirkp) != 0 and dirdes != None:
				# get two image's match
				good = []
				matches = bf.match(des,dirdes)
				for m in matches:
					if m.distance < DIST_LIMIT:
						good.append(m)

				print len(good)

				if max_match_point < len(good):
					max_match_point = len(good)
					match_filename = dirname

		output_str = filename + "ORB | match_filename:" + match_filename + " max_match_point:" +str(max_match_point)
		print output_str
		_file.write(output_str)

	elif mode == "sift":
		# init
		sift = cv2.xfeatures2d.SIFT_create()
		bf = cv2.BFMatcher()

		# get descriptors
		kp, des = sift.detectAndCompute(winelabelgray,None)

		# match images
		max_match_point=0
		DIST_LIMIT = 45
		dirs = os.listdir(imgfolderpath)
		for dirname in dirs:
			print dirname
			dirimg = myfunc.find_wine_label(imgfolderpath+"/"+dirname,False)
			height,width = dirimg.shape[:2]
			# print "height:" + str(height) + " width:" + str(width)
			if width > 0 and height > 0:
				dirgray = cv2.cvtColor(dirimg,cv2.COLOR_BGR2GRAY)
				dirkp, dirdes = sift.detectAndCompute(dirgray,None)

				if len(dirkp) != 0 and dirdes != None:
					matches = bf.knnMatch(des,dirdes, k=2)
					good = []
					for m_n in matches:
						if len(m_n) != 2:
							continue
						(m,n) = m_n
						if m.distance < 0.75*n.distance:
							good.append(m)

					print len(good)

					if max_match_point < len(good):
						max_match_point = len(good)
						match_filename = dirname

		output_str = filename + "SIFT |match_filename:" + match_filename + " max_match_point:" +str(max_match_point)
		print output_str
		_file.write(output_str)

	elif mode == "surf":
		#init
		surf = cv2.xfeatures2d.SURF_create()
		bf = cv2.BFMatcher()

		# get descriptors
		kp, des = surf.detectAndCompute(winelabelgray,None)

		# match images
		max_match_point=0
		DIST_LIMIT = 45
		dirs = os.listdir(imgfolderpath)
		for dirname in dirs:
			print dirname
			dirimg = myfunc.find_wine_label(imgfolderpath+"/"+dirname,False)
			height,width = dirimg.shape[:2]
			# print "height:" + str(height) + " width:" + str(width)
			if width > 0 and height > 0:
				dirgray = cv2.cvtColor(dirimg,cv2.COLOR_BGR2GRAY)
				dirkp, dirdes = surf.detectAndCompute(dirgray,None)

				if len(dirkp) != 0 and dirdes != None:
					matches = bf.knnMatch(des,dirdes, k=2)
					good = []
					for m_n in matches:
						if len(m_n) != 2:
							continue
						(m,n) = m_n
						if m.distance < 0.75*n.distance:
							good.append(m)

					print len(good)

					if max_match_point < len(good):
						max_match_point = len(good)
						match_filename = dirname

		output_str = filename + "SURF | match_filename:" + match_filename + " max_match_point:" +str(max_match_point)
		print output_str
		_file.write(output_str)
		_file.close()

	print "====="+filename+" END====="
else:
	print "argv1: filepath, argv2: imgfolderapth argv3: sift/surf/orb"

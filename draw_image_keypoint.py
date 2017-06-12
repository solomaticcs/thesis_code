import sys
import myfunc
import cv2
import os
import numpy as np

if len(sys.argv) > 1:
	testfilepath = sys.argv[1]
	filename = myfunc.path_leaf(testfilepath)
	print filename

	# create directory
	output_directory="./draw_image_keypoint/" + filename;
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)

	""" init """
	orb = cv2.ORB_create()
	sift = cv2.xfeatures2d.SIFT_create()
	surf = cv2.xfeatures2d.SURF_create()

	test_winelabel = myfunc.find_wine_label_v2(testfilepath,False)
	height, width = test_winelabel.shape[:2]
	if width < 250 or height < 250:
		# use origin image
		test_winelabel = cv2.imread(testfilepath)

	test_winelabelgray = cv2.cvtColor(test_winelabel,cv2.COLOR_BGR2GRAY)
	orb_kp, orb_des = orb.detectAndCompute(test_winelabelgray,None)
	sift_kp, sift_des = sift.detectAndCompute(test_winelabelgray,None)
	surf_kp, surf_des = surf.detectAndCompute(test_winelabelgray,None)
	orb_kp_draw_kp = test_winelabelgray.copy()
	sift_kp_draw_kp = test_winelabelgray.copy()
	surf_kp_draw_kp = test_winelabelgray.copy()
	# myfunc.draw_keypoints(orb_kp_draw_kp,orb_kp,(0,255,255))
	# myfunc.draw_keypoints(sift_kp_draw_kp,sift_kp,(0,255,255))
	# myfunc.draw_keypoints(surf_kp_draw_kp,surf_kp,(0,255,255))
	dummy = np.zeros((1,1))
	orb_kp_draw_kp = cv2.drawKeypoints(orb_kp_draw_kp, orb_kp,dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	sift_kp_draw_kp = cv2.drawKeypoints(sift_kp_draw_kp, sift_kp,dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	surf_kp_draw_kp = cv2.drawKeypoints(surf_kp_draw_kp, surf_kp,dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite(output_directory+ "/img.png", test_winelabel)
	cv2.imwrite(output_directory+ "/gray.png", test_winelabelgray)
	cv2.imwrite(output_directory+ "/orb_kp.png", orb_kp_draw_kp)
	cv2.imwrite(output_directory+ "/sift_kp.png", sift_kp_draw_kp)
	cv2.imwrite(output_directory + "/surf_kp.png", surf_kp_draw_kp)
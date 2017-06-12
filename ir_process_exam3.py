import ntpath
import os
import sys
from PIL import Image 
import pytesseract
import cv2
import myfunc
import re
import numpy as np
import time
import xlsxwriter

def ir(testfilepath):
	# global variable
	global orb,sift,surf,bf1,bf2,orb_des_dist,sift_des_dist,surf_des_dist,worksheet,row

	# get filename
	testfilename=myfunc.path_leaf(testfilepath)
	# get file path's wineid
	testwineid=myfunc.path_name(testfilename)
	# get file number
	testnumber=myfunc.path_number(testfilename)
	print "testfilename:" + testfilename + " testwineid:" + testwineid + " testnumber:" + testnumber

	row+=1
	cell=0
	# write "Image Name"
	worksheet.write(row,cell,testwineid)
	cell+=1

	""" get test wine label orb, sift, surf description"""
	# start the timer
	wine_seg_start_time = time.time()

	test_winelabel = myfunc.find_wine_label_v2(testfilepath,False)
	height, width = test_winelabel.shape[:2]
	# if width < 250 or height < 250:
	# 	# use origin image
	# 	test_winelabel = cv2.imread(testfilepath)

	# end the timer
	wine_seg_end_time = time.time()
	# cost time
	wine_cost_time = wine_seg_end_time - wine_seg_start_time
	# write "Wine Label Segmentation Time"
	worksheet.write(row,cell,wine_cost_time)
	cell+=1

	# convert image to gray
	if height > 0 and width > 0:
		test_winelabelgray = cv2.cvtColor(test_winelabel,cv2.COLOR_BGR2GRAY)

	if height > 0 and width > 0:
		# start the timer
		orb_des_extract_start_time = time.time()
		# extract orb description
		kp, orb_des = orb.detectAndCompute(test_winelabelgray,None)
		# end the timer
		orb_des_extract_end_time = time.time()
		# cost time
		orb_des_extract_cost_time = orb_des_extract_end_time - orb_des_extract_start_time
	else: 
		orb_des = np.zeros(0)
		orb_des_extract_cost_time = 0

	# write "Extract ORB Description Time"
	worksheet.write(row,cell,orb_des_extract_cost_time)
	cell+=1

	if height > 0 and width > 0:
		# start the timer
		sift_des_extract_start_time = time.time()
		# extract sift description
		kp, sift_des = sift.detectAndCompute(test_winelabelgray,None)
		# end the timer
		sift_des_extract_end_time = time.time()
		# cost time
		sift_des_extract_cost_time = sift_des_extract_end_time - sift_des_extract_start_time
	else:
		sift_des = np.zeros(0)
		sift_des_extract_cost_time = 0

	# write "Extract SIFT Description Time"
	worksheet.write(row,cell,sift_des_extract_cost_time)
	cell+=1

	if height > 0 and width > 0:
		# start the timer
		surf_des_extract_start_time = time.time()
		# extract surf description
		kp, surf_des = surf.detectAndCompute(test_winelabelgray,None)
		# end the timer
		surf_des_extract_end_time = time.time()
		# cost time
		surf_des_extract_cost_time = surf_des_extract_end_time - surf_des_extract_start_time
	else:
		surf_des = np.zeros(0)
		surf_des_extract_cost_time = 0

	# write "Extract SIFT Description Time"
	worksheet.write(row,cell,surf_des_extract_cost_time)
	cell+=1

	# print "orb_des des len:" + str(len(orb_des))
	# print "sift_des des len:" + str(len(sift_des))
	# print "surf_des des len:" + str(len(surf_des))

	""" Match ORB Features """
	DIST_LIMIT=45

	orb_match_wineid = "none"
	orb_max_match_point=0
	print testwineid + " START ORB"
	# start the timer
	orb_start_time = time.time()
	# match all of ORB Features
	for wineid, des in orb_des_dist.iteritems():
		if wineid==testwineid:
			continue

		if orb_des != None:
			if len(orb_des) > 0 and len(des) > 0:
				good = []
				matches = bf1.match(orb_des,des)
				for m in matches:
					if m.distance < DIST_LIMIT:
						good.append(m)

				if orb_max_match_point < len(good):
					orb_max_match_point = len(good)
					orb_match_wineid = wineid
	# stop the timer
	orb_end_time = time.time()
	# cost time
	orb_cost_time = orb_end_time - orb_start_time
	# write "ORB Match FileName"
	worksheet.write(row,cell,orb_match_wineid)
	cell+=1
	# write "ORB Max Match Point"
	worksheet.write(row,cell,orb_max_match_point)
	cell+=1
	# write "Correct"
	if orb_match_wineid != "none" and myfunc.path_number(orb_match_wineid)==testnumber:
		worksheet.write(row,cell,1)
	else:
		worksheet.write(row,cell,0)
	cell+=1
	# write "Cost time"
	worksheet.write(row,cell,orb_cost_time)
	cell+=1

	""" Match SIFT Features """
	sift_match_wineid = "none"
	sift_max_match_point=0
	print testwineid + " START SIFT"
	# start the timer
	sift_start_time = time.time()
	# match all of SIFT Features
	for wineid, des in sift_des_dist.iteritems():
		if wineid==testwineid:
			continue

		if sift_des != None:
			if len(sift_des) > 0 and len(des) >0:
				matches = bf2.knnMatch(sift_des,des, k=2)
				good = []
				for m_n in matches:
					if len(m_n) != 2:
						continue
					(m,n) = m_n
					if m.distance < 0.75*n.distance:
						good.append(m)

				if sift_max_match_point < len(good):
					sift_max_match_point = len(good)
					sift_match_wineid = wineid
	# stop the timer
	sift_end_time = time.time()
	# cost time
	sift_cost_time = sift_end_time - sift_start_time
	# write "SIFT Match FileName"
	worksheet.write(row,cell,sift_match_wineid)
	cell+=1
	# write "SIFT Max Match Point"
	worksheet.write(row,cell,sift_max_match_point)
	cell+=1
	# write "Correct"
	if sift_match_wineid != "none" and myfunc.path_number(sift_match_wineid)==testnumber:
		worksheet.write(row,cell,1)
	else:
		worksheet.write(row,cell,0)
	cell+=1
	# write "Cost time"
	worksheet.write(row,cell,sift_cost_time)
	cell+=1

	""" Match SURF Features """
	surf_match_wineid = "none"
	surf_max_match_point=0
	print testwineid + " START SURF"
	# start the timer
	surf_start_time = time.time()
	# match all of SURF Features
	for wineid, des in surf_des_dist.iteritems():
		if wineid==testwineid:
			continue

		if surf_des != None:
			if len(surf_des) > 0 and len(des) > 0:
				matches = bf2.knnMatch(surf_des,des, k=2)
				good = []
				for m_n in matches:
					if len(m_n) != 2:
						continue
					(m,n) = m_n
					if m.distance < 0.75*n.distance:
						good.append(m)

				if surf_max_match_point < len(good):
					surf_max_match_point = len(good)
					surf_match_wineid = wineid
	# stop the timer
	surf_end_time = time.time()
	# cost time
	surf_cost_time = surf_end_time - surf_start_time
	# write "SURF Match FileName"
	worksheet.write(row,cell,surf_match_wineid)
	cell+=1
	# write "SURF Max Match Point"
	worksheet.write(row,cell,surf_max_match_point)
	cell+=1
	# write "Correct"
	if surf_match_wineid != "none" and myfunc.path_number(surf_match_wineid)==testnumber:
		worksheet.write(row,cell,1)
	else:
		worksheet.write(row,cell,0)
	cell+=1
	# write "Cost time"
	worksheet.write(row,cell,surf_cost_time)
	cell+=1
def main():
	if len(sys.argv)>2:
		# global variable
		global orb,sift,surf,bf1,bf2,orb_des_dist,sift_des_dist,surf_des_dist,worksheet,row

		# test folde path
		testfolderpath=sys.argv[1]
		# train description folder path
		traindesfolderpath=sys.argv[2]
		# output folder path
		outputfolderpath=sys.argv[3]
		# create directory
		if not os.path.exists(outputfolderpath):
			os.makedirs(outputfolderpath)
		print testfolderpath
		print traindesfolderpath
		print outputfolderpath

		""" init """
		orb = cv2.ORB_create()
		sift = cv2.xfeatures2d.SIFT_create()
		surf = cv2.xfeatures2d.SURF_create()
		bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		bf2 = cv2.BFMatcher()

		""" get train's description """
		orb_des_dist = {}
		sift_des_dist = {}
		surf_des_dist = {}
		files = os.listdir(traindesfolderpath)
		ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
		for filename in ordered_files:
			desfilepath = traindesfolderpath + "/" + filename
			basename=myfunc.path_name(filename)
			wineid=basename.split("_")[0]
			method=basename.split("_")[1]
			if method == "orb":
				orb_des_dist[wineid]=np.load(desfilepath)
			elif method == "sift":
				sift_des_dist[wineid]=np.load(desfilepath)
			elif method == "surf":
				surf_des_dist[wineid]=np.load(desfilepath)

		print "orb_des_dist len:" +str(len(orb_des_dist))
		print "sift_des_dist len:" + str(len(sift_des_dist))
		print "surf_des_dist len:"+str(len(surf_des_dist))

		""" create xlsxwriter """
		# Create an new Excel file and add a worksheet.
		workbook = xlsxwriter.Workbook(outputfolderpath + "/wine_ir_exam.xlsx")
		worksheet = workbook.add_worksheet()
		titles=["Image Name","Wine Label Segmentation Time","Extract ORB Description Time","Extract SIFT Description Time","Extract SURF Description Time","ORB Match FileName","ORB Max Match Point","Correct","Cost time","SIFT Match FileName","SIFT Max Match Point","Correct","Cost time","SURF Match FileName","SURF Max Match Point","Correct","Cost time"]
		row=0
		cell=0
		for title in titles:
			worksheet.write(row,cell,title)
			cell+=1

		files = os.listdir(testfolderpath)
		ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
		for filename in ordered_files:
			print filename
			ir(testfolderpath+"/"+filename)

		# close workbook
		workbook.close()
	else:
		print "error"
main()
import ntpath
import os
import sys
from PIL import Image 
import pytesseract
import cv2
import re
import json
import numpy as np

#------------------------system------------------------

# get filename
def path_leaf(path):
	head, tail = ntpath.split(path)
	return tail or ntpath.basename(head)

# get parent
def path_parent(path):
	head, tail = ntpath.split(path)
	return head

# get filename without extension
def path_name(filename):
	return os.path.splitext(filename)[0]

# get filename extension
def path_extension(filename):
	return os.path.splitext(filename)[1]

# get filename number
def path_number(filename):
	return re.findall('\d+',filename)[0]

#----------------------------------------------------------

#------------------------function----------------------

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

def draw_keypoints(vis, keypoints, color):
	print 
	for kp in keypoints:
		x, y = kp.pt
		cv2.circle(vis, (int(x), int(y)), 50, color)

def drawMatches(img1, kp1, img2, kp2, matches):
	"""
	My own implementation of cv2.drawMatches as OpenCV 2.4.9
	does not have this function available but it's supported in
	OpenCV 3.0.0

	This function takes in two images with their associated 
	keypoints, as well as a list of DMatch data structure (matches) 
	that contains which keypoints matched in which images.

	An image will be produced where a montage is shown with
	the first image followed by the second image beside it.

	Keypoints are delineated with circles, while lines are connected
	between matching keypoints.

	img1,img2 - Grayscale images
	kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
	detection algorithms
	matches - A list of matches of corresponding keypoints through any
	OpenCV keypoint matching algorithm
	"""

	# Create a new output image that concatenates the two images together
	# (a.k.a) a montage
	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]

	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

	# Place the first image to the left
	out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

	# Place the next image to the right of it
	out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

	# For each pair of points we have between both images
	# draw circles, then connect a line between them
	for mat in matches:

		# Get the matching keypoints for each of the images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		# x - columns
		# y - rows
		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		# Draw a small circle at both co-ordinates
		# radius 4
		# colour blue
		# thickness = 1
		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

		# Draw a line in between the two points
		# thickness = 1
		# colour blue
		cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

	return out

# save descriptor
def save_des(folderpath,name, des):
	writepath = folderpath + "/" + name + '.json'
	with open(writepath, "w") as outfile:
		json.dump(des, outfile)

def read_des(filepath,alg):
	global orb,sift,surf,bf1,bf2,orb_des_dist,sift_des_dist,surf_des_dist

	print alg

	_file = open(filepath, "r")
	data = _file.read()
	_file.close()

	data = data.replace("[", "")
	data = data.replace("]", "")
	arr_list = data.split(",")

	if alg == "orb":
		arr_list = [int(i) for i in arr_list]
		
		height = arr_list[0]
		width = arr_list[1]

		size = (h, w) = (height, width)
		des = np.zeros(size,np.uint8)

		index=2
		for i in range(0, height, 1):
			for j in range(0, width, 1):
				des[i][j] = arr_list[index]
				index+=1
	elif alg == "sift" or alg == "surf":
		arr_list = [float(i) for i in arr_list]

		height = (int)(arr_list[0])
		width = (int)(arr_list[1])

		size = (h, w) = (height, width)
		des = np.zeros(size,dtype=np.float32)

		index=2
		for i in range(0, height, 1):
			for j in range(0, width, 1):
				tmp = arr_list[index]
				des[i][j] = tmp
				index+=1

	return des

# find wine label
def find_wine_label(filepath,b):
	filename = path_leaf(filepath)

	if b:
		# create directory
		wine_directory="./winelabel_region_exam_output/" + filename;
		if not os.path.exists(wine_directory):
			os.makedirs(wine_directory)

	# get image
	img=cv2.imread(filepath)
	# convert image to gray
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# binary
	ret, binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_img.png", img)
		cv2.imwrite(wine_directory + "/" + filename + "_gray.png", gray)
		cv2.imwrite(wine_directory + "/" + filename + "_binary.png", binary)

	# get contours
	im2, contours, hierarchy = cv2.findContours(binary.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
	# copy img
	img_allccl = img.copy()
	# img draw all contours
	mat_drawrect(img_allccl,contours)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_img_allccl.png", img_allccl)

	# get max contour
	max_contour = find_max_contour(contours)
	# get rect x,y and width,height
	x,y,w,h = cv2.boundingRect(max_contour)
	# copy img
	img_drawrect = img.copy()
	# img draw rect
	img_drawrect = cv2.rectangle(img_drawrect, (x,y), (x+w,y+h), (0,0,255), 5)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_img_drawrect.png", img_drawrect)

	# create pre wine label 
	prewinelabel=img[y:y+h, x:x+w]
	# convert image to gray
	prewinelabelgray=cv2.cvtColor(prewinelabel,cv2.COLOR_BGR2GRAY)
	# binary
	ret, prewinelabelbinary=cv2.threshold(prewinelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_prewinelabel.png", prewinelabel)
		cv2.imwrite(wine_directory + "/" + filename + "_prewinelabelgray.png", prewinelabelgray)
		cv2.imwrite(wine_directory + "/" + filename + "_prewinelabelbinary.png", prewinelabelbinary)

	COLOR_BLACK = 0
	height, width = prewinelabelbinary.shape[:2]

	""" get vertical black point histogram """
	vertical_black_histogram = []
	for y in range(0,height,1):
		# print y
		cnt=0
		for x in range(0,width,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		vertical_black_histogram.append(cnt)

	# custom black point threshold
	vertical_black_histogram_threshold = width*0.8
	# continuous rows
	vertical_crows = 10

	cnt=0
	tmp_top_index=0
	winelabel_top_index=0
	winelabel_bottom_index=height-1
	for k in range(0,len(vertical_black_histogram),1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			if cnt >= vertical_crows:
				winelabel_bottom_index=k-1
			cnt=0
		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_top_index=k
			cnt=cnt+1
		if cnt >= vertical_crows:
			winelabel_top_index=tmp_top_index

	""" get horizontal black point histogram """
	horizontal_black_histogram = []
	for x in range(0,width,1):
		# print y
		cnt=0
		for y in range(0,height,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		horizontal_black_histogram.append(cnt)

	# custom black point threshold
	horizontal_black_histogram_threshold = height*0.8
	# continuous rows
	horizontal_crows = 10
	
	cnt=0
	tmp_left_index=0
	winelabel_left_index=0
	winelabel_right_index=height-1
	for k in range(0,len(horizontal_black_histogram),1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt >= horizontal_crows:
				winelabel_right_index=k-1
			cnt=0
		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_left_index=k
			cnt=cnt+1
		if cnt >= horizontal_crows:
			winelabel_left_index=tmp_left_index

	# copy img
	winelabel_rect = prewinelabel.copy()
	# img draw rect
	winelabel_rect = cv2.rectangle(winelabel_rect, (winelabel_left_index,winelabel_top_index), (winelabel_right_index,winelabel_bottom_index), (0,0,255), 5)

	winelabel = prewinelabel[winelabel_top_index:winelabel_bottom_index,winelabel_left_index:winelabel_right_index]

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_winelabel_rect.png", winelabel_rect)
		cv2.imwrite(wine_directory + "/" + filename + "_winelabel.png", winelabel)
	return winelabel

# find wine label v2
def find_wine_label_v2(filepath,b):
	filename = path_leaf(filepath)

	if b:
		# create directory
		wine_directory="./winelabel_region_exam_output_v2/" + filename;
		if not os.path.exists(wine_directory):
			os.makedirs(wine_directory)

	# get image
	img=cv2.imread(filepath)
	# convert image to gray
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# binary
	ret, binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_img.png", img)
		cv2.imwrite(wine_directory + "/" + filename + "_gray.png", gray)
		cv2.imwrite(wine_directory + "/" + filename + "_binary.png", binary)

	# get contours
	im2, contours, hierarchy = cv2.findContours(binary.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
	# copy img
	img_allccl = img.copy()
	# img draw all contours
	mat_drawrect(img_allccl,contours)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_img_allccl.png", img_allccl)

	# get max contour
	max_contour = find_max_contour(contours)
	# get rect x,y and width,height
	x,y,w,h = cv2.boundingRect(max_contour)
	# copy img
	img_drawrect = img.copy()
	# img draw rect
	img_drawrect = cv2.rectangle(img_drawrect, (x,y), (x+w,y+h), (0,0,255), 5)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_img_drawrect.png", img_drawrect)

	# create pre wine label 
	prewinelabel=img[y:y+h, x:x+w]
	# convert image to gray
	prewinelabelgray=cv2.cvtColor(prewinelabel,cv2.COLOR_BGR2GRAY)
	# binary
	ret, prewinelabelbinary=cv2.threshold(prewinelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_prewinelabel.png", prewinelabel)
		cv2.imwrite(wine_directory + "/" + filename + "_prewinelabelgray.png", prewinelabelgray)
		cv2.imwrite(wine_directory + "/" + filename + "_prewinelabelbinary.png", prewinelabelbinary)

	COLOR_BLACK = 0
	height, width = prewinelabelbinary.shape[:2]

	""" get vertical black point histogram """
	vertical_black_histogram = []
	for y in range(0,height,1):
		# print y
		cnt=0
		for x in range(0,width,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		vertical_black_histogram.append(cnt)

	# custom black point threshold
	vertical_black_histogram_threshold = width*0.8
	# continuous rows
	vertical_crows = 10
	# start position(vertical middle)
	vertical_middle_index=height/2

	cnt=0
	tmp_bottom_index=height-1
	winelabel_bottom_index=height-1
	# Run-length
	# middle to down(get bottom index)
	for k in range(vertical_middle_index,len(vertical_black_histogram),1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_bottom_index=k
			cnt=cnt+1

		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			cnt=0

		if cnt >= vertical_crows:
			winelabel_bottom_index=tmp_bottom_index
			break
	cnt=0
	tmp_top_index=0
	winelabel_top_index=0
	# middle to up (get top index)
	for k in range(vertical_middle_index,-1,-1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_top_index=k
			cnt=cnt+1

		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			cnt=0

		if cnt >= vertical_crows:
			winelabel_top_index=tmp_top_index
			break

	""" get horizontal black point histogram """
	horizontal_black_histogram = []
	for x in range(0,width,1):
		# print y
		cnt=0
		for y in range(0,height,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		horizontal_black_histogram.append(cnt)

	# custom black point threshold
	horizontal_black_histogram_threshold = height*0.8
	# continuous rows
	horizontal_crows = 10
	# start position(horizontal middle)
	horizontal_middle_index=width/2

	cnt=0
	tmp_right_index=width-1
	winelabel_right_index=width-1
	# Run-length
	# middle to down(get width index)
	for k in range(horizontal_middle_index,len(horizontal_black_histogram),1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_right_index=k
			cnt=cnt+1

		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			cnt=0

		if cnt >= horizontal_crows:
			winelabel_right_index=tmp_right_index
			break
	cnt=0
	tmp_left_index=0
	winelabel_left_index=0
	# middle to up (get left index)
	for k in range(horizontal_middle_index,-1,-1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_left_index=k
			cnt=cnt+1

		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			cnt=0

		if cnt >= horizontal_crows:
			winelabel_left_index=tmp_left_index
			break

	# copy img
	winelabel_rect = prewinelabel.copy()
	# img draw rect
	winelabel_rect = cv2.rectangle(winelabel_rect, (winelabel_left_index,winelabel_top_index), (winelabel_right_index,winelabel_bottom_index), (0,0,255), 5)
	# wine label segmentation
	winelabel = prewinelabel[winelabel_top_index:winelabel_bottom_index,winelabel_left_index:winelabel_right_index]

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_winelabel_rect.png", winelabel_rect)
		cv2.imwrite(wine_directory + "/" + filename + "_winelabel.png", winelabel)

	return winelabel

# find wine label v3
def find_wine_label_v3(filepath,b):
	filename = path_leaf(filepath)

	if b:
		# create directory
		wine_directory="./winelabel_region_exam_output_v3/" + filename;
		if not os.path.exists(wine_directory):
			os.makedirs(wine_directory)

	# get image
	img=cv2.imread(filepath)
	# convert image to gray
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# binary
	ret, binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_img.png", img)
		cv2.imwrite(wine_directory + "/" + filename + "_gray.png", gray)
		cv2.imwrite(wine_directory + "/" + filename + "_binary.png", binary)

	# get contours
	im2, contours, hierarchy = cv2.findContours(binary.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
	# copy img
	img_allccl = img.copy()
	# img draw all contours
	mat_drawrect(img_allccl,contours)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_img_allccl.png", img_allccl)

	# get max contour
	max_contour = find_max_contour(contours)
	# get rect x,y and width,height
	x,y,w,h = cv2.boundingRect(max_contour)
	# copy img
	img_drawrect = img.copy()
	# img draw rect
	img_drawrect = cv2.rectangle(img_drawrect, (x,y), (x+w,y+h), (0,0,255), 5)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_img_drawrect.png", img_drawrect)

	# create pre wine label 
	prewinelabel=img[y:y+h, x:x+w]
	# convert image to gray
	prewinelabelgray=cv2.cvtColor(prewinelabel,cv2.COLOR_BGR2GRAY)
	# binary
	ret, prewinelabelbinary=cv2.threshold(prewinelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_prewinelabel.png", prewinelabel)
		cv2.imwrite(wine_directory + "/" + filename + "_prewinelabelgray.png", prewinelabelgray)
		cv2.imwrite(wine_directory + "/" + filename + "_prewinelabelbinary.png", prewinelabelbinary)

	COLOR_BLACK = 0
	height, width = prewinelabelbinary.shape[:2]

	""" get vertical black point histogram """
	vertical_black_histogram = []
	for y in range(0,height,1):
		# print y
		cnt=0
		for x in range(0,width,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		vertical_black_histogram.append(cnt)

	# custom black point threshold
	vertical_black_histogram_threshold = width*0.8
	# continuous rows
	vertical_crows = 10

	# get winelabel top index
	cnt=0
	tmp_top_index=0
	winelabel_top_index=0
	for k in range(0,int(len(vertical_black_histogram)*0.2),1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			cnt=0
		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_top_index=k
			cnt=cnt+1
		if cnt >= vertical_crows:
			winelabel_top_index=tmp_top_index
			break

	# get winelabel bottom index
	cnt=0
	tmp_bottom_index=height-1
	winelabel_bottom_index=height-1
	for k in range(int(len(vertical_black_histogram))-1,int(len(vertical_black_histogram)*0.2),-1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			cnt=0
		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_bottom_index=k
			cnt=cnt+1
		if cnt >= vertical_crows:
			winelabel_bottom_index=tmp_bottom_index
			break

	""" get horizontal black point histogram """
	horizontal_black_histogram = []
	for x in range(0,width,1):
		# print y
		cnt=0
		for y in range(0,height,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		horizontal_black_histogram.append(cnt)

	# custom black point threshold
	horizontal_black_histogram_threshold = height*0.8
	# continuous rows
	horizontal_crows = 10
	
	# get winelabel left index
	cnt=0
	tmp_left_index=0
	winelabel_left_index=0
	for k in range(0,int(len(horizontal_black_histogram)*0.2),1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt >= horizontal_crows:
				cnt=0
		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_left_index=k
			cnt=cnt+1
		if cnt >= horizontal_crows:
			winelabel_left_index=tmp_left_index
			break

	# get winelabel right index
	cnt=0
	tmp_right_index=width-1
	winelabel_right_index=width-1
	for k in range(int(len(horizontal_black_histogram))-1,int(len(horizontal_black_histogram)*0.2),-1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt >= horizontal_crows:
				cnt=0
		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_right_index=k
			cnt=cnt+1
		if cnt >= horizontal_crows:
			winelabel_right_index=tmp_right_index
			break

	# copy img
	winelabel_rect = prewinelabel.copy()
	# img draw rect
	winelabel_rect = cv2.rectangle(winelabel_rect, (winelabel_left_index,winelabel_top_index), (winelabel_right_index,winelabel_bottom_index), (0,0,255), 5)

	winelabel = prewinelabel[winelabel_top_index:winelabel_bottom_index,winelabel_left_index:winelabel_right_index]
	winelabelgray=cv2.cvtColor(winelabel,cv2.COLOR_BGR2GRAY)
	ret, winelabelbinary=cv2.threshold(winelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	if b:
		# output
		cv2.imwrite(wine_directory + "/" + filename + "_winelabel_rect.png", winelabel_rect)
		cv2.imwrite(wine_directory + "/" + filename + "_winelabel.png", winelabel)
		cv2.imwrite(wine_directory + "/" + filename + "_winelabelbinary.png", winelabelbinary)

	return winelabel

# find max contour
def find_max_contour(contours):
	max_area = 0
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > max_area:
			max_area = area
			max_contour = contour
	return max_contour

# filter contours
def filter_contours(img_width, img_height, contours):
	new_contours=[];
	for contour in contours:
		if cv2.contourArea(contour)>50:
			x,y,w,h = cv2.boundingRect(contour)

			if h>10 and w<0.5*img_width and h<0.5*img_height:
				new_contours.append(contour)
	return new_contours

# mat --> draw rect 
def mat_drawrect(img, contours):
	for contour in contours:
		x,y,w,h = cv2.boundingRect(contour)
		cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)

def wine_segment_v1(filepath):
	filename = path_leaf(filepath)
	# get image
	img=cv2.imread(filepath)
	# convert image to gray
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# binary
	ret, binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# get contours
	im2, contours, hierarchy = cv2.findContours(binary.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
	# copy img
	img_allccl = img.copy()
	# img draw all contours
	mat_drawrect(img_allccl,contours)
	# get max contour
	max_contour = find_max_contour(contours)
	# get rect x,y and width,height
	x,y,w,h = cv2.boundingRect(max_contour)
	# copy img
	img_drawrect = img.copy()
	# img draw rect
	img_drawrect = cv2.rectangle(img_drawrect, (x,y), (x+w,y+h), (0,0,255), 5)
	# create pre wine label 
	prewinelabel=img[y:y+h, x:x+w]
	# convert image to gray
	prewinelabelgray=cv2.cvtColor(prewinelabel,cv2.COLOR_BGR2GRAY)
	# binary
	ret, prewinelabelbinary=cv2.threshold(prewinelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	COLOR_BLACK = 0
	height, width = prewinelabelbinary.shape[:2]

	""" get vertical black point histogram """
	vertical_black_histogram = []
	for y in range(0,height,1):
		# print y
		cnt=0
		for x in range(0,width,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		vertical_black_histogram.append(cnt)

	# custom black point threshold
	vertical_black_histogram_threshold = width*0.8
	# continuous rows
	vertical_crows = 10

	cnt=0
	tmp_top_index=0
	winelabel_top_index=0
	winelabel_bottom_index=height-1
	for k in range(0,len(vertical_black_histogram),1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			if cnt >= vertical_crows:
				winelabel_bottom_index=k-1
			cnt=0
		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_top_index=k
			cnt=cnt+1
		if cnt >= vertical_crows:
			winelabel_top_index=tmp_top_index

	""" get horizontal black point histogram """
	horizontal_black_histogram = []
	for x in range(0,width,1):
		# print y
		cnt=0
		for y in range(0,height,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		horizontal_black_histogram.append(cnt)

	# custom black point threshold
	horizontal_black_histogram_threshold = height*0.8
	# continuous rows
	horizontal_crows = 10
	
	cnt=0
	tmp_left_index=0
	winelabel_left_index=0
	winelabel_right_index=height-1
	for k in range(0,len(horizontal_black_histogram),1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt >= horizontal_crows:
				winelabel_right_index=k-1
			cnt=0
		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_left_index=k
			cnt=cnt+1
		if cnt >= horizontal_crows:
			winelabel_left_index=tmp_left_index

	# copy img
	winelabel_rect = prewinelabel.copy()
	# img draw rect
	winelabel_rect = cv2.rectangle(winelabel_rect, (winelabel_left_index,winelabel_top_index), (winelabel_right_index,winelabel_bottom_index), (0,0,255), 5)

	# winelabel = prewinelabel[winelabel_top_index:winelabel_bottom_index,winelabel_left_index:winelabel_right_index]

	return winelabel_rect

def wine_segment_v2(filepath):
	filename = path_leaf(filepath)

	# get image
	img=cv2.imread(filepath)
	# convert image to gray
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# binary
	ret, binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	# get contours
	im2, contours, hierarchy = cv2.findContours(binary.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
	# copy img
	img_allccl = img.copy()
	# img draw all contours
	mat_drawrect(img_allccl,contours)

	# get max contour
	max_contour = find_max_contour(contours)
	# get rect x,y and width,height
	x,y,w,h = cv2.boundingRect(max_contour)
	# copy img
	img_drawrect = img.copy()
	# img draw rect
	img_drawrect = cv2.rectangle(img_drawrect, (x,y), (x+w,y+h), (0,0,255), 5)

	# create pre wine label 
	prewinelabel=img[y:y+h, x:x+w]
	# convert image to gray
	prewinelabelgray=cv2.cvtColor(prewinelabel,cv2.COLOR_BGR2GRAY)
	# binary
	ret, prewinelabelbinary=cv2.threshold(prewinelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	COLOR_BLACK = 0
	height, width = prewinelabelbinary.shape[:2]

	""" get vertical black point histogram """
	vertical_black_histogram = []
	for y in range(0,height,1):
		# print y
		cnt=0
		for x in range(0,width,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		vertical_black_histogram.append(cnt)

	# custom black point threshold
	vertical_black_histogram_threshold = width*0.8
	# continuous rows
	vertical_crows = 10
	# start position(vertical middle)
	vertical_middle_index=height/2

	cnt=0
	tmp_bottom_index=height-1
	winelabel_bottom_index=height-1
	# Run-length
	# middle to down(get bottom index)
	for k in range(vertical_middle_index,len(vertical_black_histogram),1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_bottom_index=k
			cnt=cnt+1

		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			cnt=0

		if cnt >= vertical_crows:
			winelabel_bottom_index=tmp_bottom_index
			break
	cnt=0
	tmp_top_index=0
	winelabel_top_index=0
	# middle to up (get top index)
	for k in range(vertical_middle_index,-1,-1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_top_index=k
			cnt=cnt+1

		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			cnt=0

		if cnt >= vertical_crows:
			winelabel_top_index=tmp_top_index
			break

	""" get horizontal black point histogram """
	horizontal_black_histogram = []
	for x in range(0,width,1):
		# print y
		cnt=0
		for y in range(0,height,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		horizontal_black_histogram.append(cnt)

	# custom black point threshold
	horizontal_black_histogram_threshold = height*0.8
	# continuous rows
	horizontal_crows = 10
	# start position(horizontal middle)
	horizontal_middle_index=width/2

	cnt=0
	tmp_right_index=width-1
	winelabel_right_index=width-1
	# Run-length
	# middle to down(get width index)
	for k in range(horizontal_middle_index,len(horizontal_black_histogram),1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_right_index=k
			cnt=cnt+1

		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			cnt=0

		if cnt >= horizontal_crows:
			winelabel_right_index=tmp_right_index
			break
	cnt=0
	tmp_left_index=0
	winelabel_left_index=0
	# middle to up (get left index)
	for k in range(horizontal_middle_index,-1,-1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_left_index=k
			cnt=cnt+1

		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			cnt=0

		if cnt >= horizontal_crows:
			winelabel_left_index=tmp_left_index
			break

	# copy img
	winelabel_rect = prewinelabel.copy()
	# img draw rect
	winelabel_rect = cv2.rectangle(winelabel_rect, (winelabel_left_index,winelabel_top_index), (winelabel_right_index,winelabel_bottom_index), (0,0,255), 5)
	# wine label segmentation
	# winelabel = prewinelabel[winelabel_top_index:winelabel_bottom_index,winelabel_left_index:winelabel_right_index]

	return winelabel_rect

def wine_segment_v3(filepath):
	filename = path_leaf(filepath)

	# get image
	img=cv2.imread(filepath)
	# convert image to gray
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# binary
	ret, binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	# get contours
	im2, contours, hierarchy = cv2.findContours(binary.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
	# copy img
	img_allccl = img.copy()
	# img draw all contours
	mat_drawrect(img_allccl,contours)

	# get max contour
	max_contour = find_max_contour(contours)
	# get rect x,y and width,height
	x,y,w,h = cv2.boundingRect(max_contour)
	# copy img
	img_drawrect = img.copy()
	# img draw rect
	img_drawrect = cv2.rectangle(img_drawrect, (x,y), (x+w,y+h), (0,0,255), 5)

	# create pre wine label 
	prewinelabel=img[y:y+h, x:x+w]
	# convert image to gray
	prewinelabelgray=cv2.cvtColor(prewinelabel,cv2.COLOR_BGR2GRAY)
	# binary
	ret, prewinelabelbinary=cv2.threshold(prewinelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	COLOR_BLACK = 0
	height, width = prewinelabelbinary.shape[:2]

	""" get vertical black point histogram """
	vertical_black_histogram = []
	for y in range(0,height,1):
		# print y
		cnt=0
		for x in range(0,width,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		vertical_black_histogram.append(cnt)

	# custom black point threshold
	vertical_black_histogram_threshold = width*0.8
	# continuous rows
	vertical_crows = 10

	# get winelabel top index
	cnt=0
	tmp_top_index=0
	winelabel_top_index=0
	for k in range(0,int(len(vertical_black_histogram)*0.2),1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			cnt=0
		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_top_index=k
			cnt=cnt+1
		if cnt >= vertical_crows:
			winelabel_top_index=tmp_top_index
			break

	# get winelabel bottom index
	cnt=0
	tmp_bottom_index=height-1
	winelabel_bottom_index=height-1
	for k in range(int(len(vertical_black_histogram))-1,int(len(vertical_black_histogram)*0.2),-1):
		if vertical_black_histogram[k]>vertical_black_histogram_threshold:
			cnt=0
		if vertical_black_histogram[k]<vertical_black_histogram_threshold:
			if cnt == 0:
				tmp_bottom_index=k
			cnt=cnt+1
		if cnt >= vertical_crows:
			winelabel_bottom_index=tmp_bottom_index
			break

	""" get horizontal black point histogram """
	horizontal_black_histogram = []
	for x in range(0,width,1):
		# print y
		cnt=0
		for y in range(0,height,1):
			if prewinelabelbinary[y,x] == COLOR_BLACK:
				cnt=cnt+1
		horizontal_black_histogram.append(cnt)

	# custom black point threshold
	horizontal_black_histogram_threshold = height*0.8
	# continuous rows
	horizontal_crows = 10
	
	# get winelabel left index
	cnt=0
	tmp_left_index=0
	winelabel_left_index=0
	for k in range(0,int(len(horizontal_black_histogram)*0.2),1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt >= horizontal_crows:
				cnt=0
		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_left_index=k
			cnt=cnt+1
		if cnt >= horizontal_crows:
			winelabel_left_index=tmp_left_index
			break

	# get winelabel right index
	cnt=0
	tmp_right_index=width-1
	winelabel_right_index=width-1
	for k in range(int(len(horizontal_black_histogram))-1,int(len(horizontal_black_histogram)*0.2),-1):
		if horizontal_black_histogram[k]>horizontal_black_histogram_threshold:
			if cnt >= horizontal_crows:
				cnt=0
		if horizontal_black_histogram[k]<horizontal_black_histogram_threshold:
			if cnt == 0:
				tmp_right_index=k
			cnt=cnt+1
		if cnt >= horizontal_crows:
			winelabel_right_index=tmp_right_index
			break

	# copy img
	winelabel_rect = prewinelabel.copy()
	# img draw rect
	winelabel_rect = cv2.rectangle(winelabel_rect, (winelabel_left_index,winelabel_top_index), (winelabel_right_index,winelabel_bottom_index), (0,0,255), 5)

	# winelabel = prewinelabel[winelabel_top_index:winelabel_bottom_index,winelabel_left_index:winelabel_right_index]
	# winelabelgray=cv2.cvtColor(winelabel,cv2.COLOR_BGR2GRAY)
	# ret, winelabelbinary=cv2.threshold(winelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	return winelabel_rect

#----------------------------------------------------------
import matplotlib.pyplot as plt
import ntpath
import os
import sys
import Image 
import pytesseract
import cv2

#------------------------system------------------------

# get filename
def path_leaf(path):
	head, tail = ntpath.split(path)
	return tail or ntpath.basename(head)

#----------------------------------------------------------

#------------------------function----------------------

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

#----------------------------------------------------------

if len(sys.argv)>1:
	for i in range(1, len(sys.argv), 1):
		filepath=sys.argv[i]
		print filepath
		filename=path_leaf(filepath)
		print filename

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

		# get contours
		im2, contours, hierarchy = cv2.findContours(binary.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

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

		""" get horizontal black point histogram """
		horizontal_black_histogram = []
		for x in range(0,width,1):
			# print y
			cnt=0
			for y in range(0,height,1):
				if prewinelabelbinary[y,x] == COLOR_BLACK:
					cnt=cnt+1
			horizontal_black_histogram.append(cnt)

		""" show histogram """
		x=[]
		for i in range(1,height+1):
			x.append(i)
		plt.plot(x,vertical_black_histogram)
		plt.xlabel("y position")
		plt.ylabel("count")
		plt.title('Vertical Black Histogram')
		plt.legend()
		plt.show()

		x=[]
		for i in range(1,width+1):
			x.append(i)
		plt.plot(x,horizontal_black_histogram)
		plt.xlabel("x position")
		plt.ylabel("count")
		plt.title('Horizontal Black Histogram')
		plt.legend()
		plt.show()

		print "====="+filename+" END====="
else:
	print 'please input image file arguments'
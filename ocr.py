import ntpath
import os
import sys
import Image 
import pytesseract
import cv2
import myfunc

if len(sys.argv)>1:
	filepath=sys.argv[1]
	print filepath
	filename=myfunc.path_leaf(filepath)
	print filename

	# create directory
	ocr_directory="./ocr_output/" + filename;
	if not os.path.exists(ocr_directory):
		os.makedirs(ocr_directory)

	# get wine label
	winelabel=myfunc.find_wine_label_v2(filepath,True)
	# convert image to gray
	winelabelgray=cv2.cvtColor(winelabel,cv2.COLOR_BGR2GRAY)
	print winelabelgray.shape
	# binary
	ret, winelabelbinary=cv2.threshold(winelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# # output
	# cv2.imwrite(ocr_directory + "/winelabel.png", winelabel)
	# cv2.imwrite(ocr_directory + "/winelabelgray.png", winelabelgray)
	# cv2.imwrite(ocr_directory + "/winelabelbinary.png", winelabelbinary)

	""" Text CCL """
	# copy mat
	ccl = winelabel.copy()
	# get contours
	im2, contours, hierarchy = cv2.findContours(winelabelbinary.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	print "contours len:"+str(len(contours))
	# width,height
	height, width = winelabelbinary.shape[:2]
	# filter contours
	filter_contours = myfunc.filter_contours(width,height,contours)
	print "filter_contours len:"+str(len(filter_contours))
	# filter contours > 0
	if len(filter_contours)>0:
		# # draw rect
		# myfunc.mat_drawrect(ccl, filter_contours)
		# # output
		# cv2.imwrite(ocr_directory + "/ccl.png", ccl)	
		
		"""text line merge"""
		# top down
		ccl_bounding=[]
		tmp={"top":0,"down":0}
		tmpcnt=0
		for y in range(0,height,1):
			cnt=0
			for contour in filter_contours:
				rect_x,rect_y,rect_w,rect_h = cv2.boundingRect(contour)
				if y>=rect_y and y <= (rect_y + rect_h):
					if cnt==0 and tmpcnt==0:
						tmp["top"]=y
					cnt=cnt+1

			# have component
			if cnt>0:
				tmpcnt=cnt;

			# no any component
			if cnt==0:
				if tmpcnt>0:
					#
					tmp["bottom"]=y-1
					#
					ccl_bounding.append(tmp)
					# init
					tmp={"top":0,"down":0}
					tmpcnt=0
		
		print "ccl_bounding: "+str(len(ccl_bounding))

		# _file = open(ocr_directory+"/"+filename+"_ocr_text.txt", "w")
		
		# for k in range(0,len(ccl_bounding),1):
		# 	print "top:" + str(ccl_bounding[k]["top"]) + " bottom:" + str(ccl_bounding[k]["bottom"])
		# 	# image
		# 	cutimage = winelabel[ccl_bounding[k]["top"]:ccl_bounding[k]["bottom"],0:width]
		# 	# gray
		# 	cutimagegray =  cv2.cvtColor(cutimage,cv2.COLOR_BGR2GRAY)	
		# 	# binary
		# 	ret, cutimagebinary=cv2.threshold(cutimagegray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		# 	# output
		# 	cv2.imwrite(ocr_directory + "/ccl_first_region_"+str(k)+".png", cutimage)
		# 	cv2.imwrite(ocr_directory + "/ccl_first_region_"+str(k)+"_gray.png",cutimagegray)
		# 	cv2.imwrite(ocr_directory + "/ccl_first_region_"+str(k)+"_binary.png",cutimagebinary)

		# 	# ocr text
		# 	iocr =  pytesseract.image_to_string(Image.open(ocr_directory + "/ccl_first_region_"+str(k)+".png"))
		# 	gocr = pytesseract.image_to_string(Image.open(ocr_directory + "/ccl_first_region_"+str(k)+"_gray.png"))
		# 	bocr = pytesseract.image_to_string(Image.open(ocr_directory + "/ccl_first_region_"+str(k)+"_binary.png"))
		# 	iocr_text =  "img ocr:" + iocr
		# 	gocr_text =  "gray ocr:" + gocr
		# 	bocr_text =  "binary ocr:" + bocr
		# 	_file.write("=====ccl_first_region_"+str(k)+"=====\n")
		# 	_file.write(iocr_text+"\n")
		# 	_file.write(gocr_text+"\n")
		# 	_file.write(bocr_text+"\n\n")
			
		# _file.close()
	print "====="+filename+" END====="
else:
	print "error"
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

		print "ret:"+str(ret)

		# get grayscale width,height
		height, width = gray.shape
		# init histogram
		gray_histogram = []
		for k in range(256):
			gray_histogram.append(0)

		# get histogram 
		for y in range(0,height):
			for x in range(0,width):
				gray_histogram[gray[y][x]]=gray_histogram[gray[y][x]]+1

		#show histogram
		x=[]
		for i in range(256):
			x.append(i)
		plt.plot(x,gray_histogram)
		plt.xlabel("gray value")
		plt.ylabel("count")
		plt.title('Gray Histogram')
		plt.legend()
		plt.show()
		
		print "====="+filename+" END====="
else:
	print 'please input image file arguments'
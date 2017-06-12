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
	method=int(sys.argv[2])
	print method
	if method==1:
		winelabel = myfunc.find_wine_label(filepath, True)
	elif method==2:
		winelabel = myfunc.find_wine_label_v2(filepath, True)
	elif method==3:
		winelabel = myfunc.find_wine_label_v3(filepath, True)
else:
	print 'please input image file arguments'
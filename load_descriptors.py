import ntpath
import os
import sys
import cv2
import myfunc
import re
import numpy as np
import json

def main():
	# get des folder
	desfolderpath=sys.argv[1]
	# get image folder files name
	files = os.listdir(desfolderpath)
	# order files
	ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
	for filename in ordered_files:
		filepath = desfolderpath + "/" + filename
		print filepath
		des = np.load(filepath)
		print len(des)

main()
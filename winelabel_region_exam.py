import ntpath
import os
import sys
import Image 
import pytesseract
import cv2
import myfunc
import time
import xlsxwriter
import re

def main():
	if len(sys.argv)>1:
		imgfolderpath=sys.argv[1]
		method=int(sys.argv[2])
		outputfolderpath=sys.argv[3]
		if not os.path.exists(outputfolderpath):
			os.makedirs(outputfolderpath)
		""" create xlsxwriter """
		# Create an new Excel file and add a worksheet.
		workbook = xlsxwriter.Workbook(outputfolderpath+"/wine_region_exam_"+str(method)+".xlsx")
		worksheet = workbook.add_worksheet()
		titles=["Image Name","Wine Label Segmentation Time"]
		row=0
		cell=0
		for title in titles:
			worksheet.write(row,cell,title)
			cell+=1

		row=1
		
		files = os.listdir(imgfolderpath)
		ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
		for filename in ordered_files:
			cell=0
			wineid = myfunc.path_name(filename)
			start_time = time.time()
			if method==1:
				winelabel = myfunc.find_wine_label(imgfolderpath+"/"+filename, True)
			elif method==2:
				winelabel = myfunc.find_wine_label_v2(imgfolderpath+"/"+filename, True)
			elif method==3:
				winelabel = myfunc.find_wine_label_v3(imgfolderpath+"/"+filename, True)
			end_time = time.time()
			cost_time = end_time - start_time

			# write "Wine id"
			worksheet.write(row,cell,wineid)
			cell+=1
			# write "cost time"
			worksheet.write(row,cell,cost_time)
			cell+=1

			row+=1
			print "====="+filename+" END====="

		# close workbook
		workbook.close()
	else:
		print 'please input image file arguments'

main()
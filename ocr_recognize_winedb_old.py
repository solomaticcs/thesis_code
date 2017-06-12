import ntpath
import os
import sys
from PIL import Image 
import pytesseract
import cv2
import myfunc
import re
import csv, sqlite3
import xlsxwriter

def ocr(filepath):
	filename=myfunc.path_leaf(filepath)
	print filename

	# create directory
	ocr_directory = output_directory + "/" + filename;
	if not os.path.exists(ocr_directory):
		os.makedirs(ocr_directory)

	winelabel=myfunc.find_wine_label_v2(filepath,False)
	height, width = winelabel.shape[:2]
	if width < 250 or height < 250:
		# use origin image
		winelabel = cv2.imread(filepath)

	# convert image to gray
	winelabelgray=cv2.cvtColor(winelabel,cv2.COLOR_BGR2GRAY)
	# binary
	ret, winelabelbinary=cv2.threshold(winelabelgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# output
	cv2.imwrite(ocr_directory + "/winelabel.png", winelabel)
	cv2.imwrite(ocr_directory + "/winelabelgray.png", winelabelgray)
	cv2.imwrite(ocr_directory + "/winelabelbinary.png", winelabelbinary)

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


	ocr_recognized_text = []
	# filter contours > 0
	if len(filter_contours)>0:
		# draw rect
		myfunc.mat_drawrect(ccl, filter_contours)
		# output
		cv2.imwrite(ocr_directory + "/ccl.png", ccl)	
		
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

		
		for k in range(0,len(ccl_bounding),1):
			print "top:" + str(ccl_bounding[k]["top"]) + " bottom:" + str(ccl_bounding[k]["bottom"])
			# image
			cutimage = winelabel[ccl_bounding[k]["top"]:ccl_bounding[k]["bottom"],0:width]
			# gray
			cutimagegray =  cv2.cvtColor(cutimage,cv2.COLOR_BGR2GRAY)	
			# binary
			ret, cutimagebinary=cv2.threshold(cutimagegray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# output
			cv2.imwrite(ocr_directory + "/text_region_"+str(k)+".png", cutimage)
			cv2.imwrite(ocr_directory + "/text_region_"+str(k)+"_gray.png",cutimagegray)
			cv2.imwrite(ocr_directory + "/text_region_"+str(k)+"_binary.png",cutimagebinary)
			# ocr text
			bocr = pytesseract.image_to_string(Image.open(ocr_directory + "/text_region_"+str(k)+"_binary.png"))
			if bocr.strip() != "":
				ocr_recognized_text.append(bocr)
	else:
		print "no contour"

	return ocr_recognized_text

def main():
	if len(sys.argv)>1:
		global output_directory

		# create directory
		output_directory="./ocr_recognized_output/"
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)

		""" create xlsxwriter """
		# Create an new Excel file and add a worksheet.
		workbook = xlsxwriter.Workbook(output_directory + "/wine_ocr_recognize_exam.xlsx")
		worksheet = workbook.add_worksheet()
		titles=["Image Name", "Is Hit?"]
		row=0
		cell=0
		for title in titles:
			worksheet.write(row,cell,title)
			cell+=1
		worksheet2 = workbook.add_worksheet()
		titles2=["Image Name","Recognize Text", "Hit Wine ID"]
		row2=0
		cell2=0
		for title2 in titles2:
			worksheet2.write(row2,cell2,title2)
			cell2+=1

		# sqlite
		con = sqlite3.connect("wine.sqlite")
		cur = con.cursor()

		row=1
		row2=1

		imgfolderpath=sys.argv[1]
		files = os.listdir(imgfolderpath)
		ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
		for filename in ordered_files:
			filepath = imgfolderpath+"/"+filename
			print filepath
			wineid = myfunc.path_name(filename)
			print wineid
			ocr_recognized_text = ocr(filepath)
			if len(ocr_recognized_text) > 0:
				is_hit = 0
				for k in range(0, len(ocr_recognized_text),1):
					print "OCR Recognized Text: " +ocr_recognized_text[k]
					sql = """SELECT `ID`,`Name`,`Rating`,`PriceTWD`,`Winery`,`Grapes`,
							`WineRegion`,`RegionalStyle`,`Country`,`FoodPairing`,`Review`,`ImageURL`,`URL` 
							FROM `wine-data-v2-db` WHERE `Name` LIKE ? """
					cur.execute(sql, ("%"+ocr_recognized_text[k].decode("utf8")+"%", ))
					datas = cur.fetchall()
					# print datas
					if datas is None or len(datas) == 0:
						print "no data"
						cell2 = 0
						# write "Image Name"
						worksheet2.write(row2, cell2, wineid)
						cell2+=1
						# write "Recognize Text"
						worksheet2.write(row2, cell2, "")
						cell2+=1
						# write "Hit Wine ID"
						worksheet2.write(row2, cell2, "")
						cell2+=1
						row2+=1
					else:
						is_hit =1

						hit_wine_id_record = []
						hit_wine_Information_record = []
						for data in datas:
							hit_wine_id_record.append(data[0])

						print hit_wine_id_record

						cell2 = 0
						# write "Image Name"
						worksheet2.write(row2, cell2, wineid)
						cell2+=1
						# write "Recognize Text"
						worksheet2.write(row2, cell2, ocr_recognized_text[k])
						cell2+=1
						# write "Hit Wine ID"
						hit_wine_id_record_str = ""
						for theID in hit_wine_id_record:
							hit_wine_id_record_str += str(theID) + ", "
						worksheet2.write(row2, cell2, hit_wine_id_record_str)
						cell2+=1
						row2+=1
				cell = 0
				# write "Image Name"
				worksheet.write(row, cell, wineid)
				cell+=1
				# write "Is Hit?"
				worksheet.write(row, cell, is_hit)
				cell+=1
				row+=1
			else:
				cell = 0
				# write "Image Name"
				worksheet.write(row, cell, wineid)
				cell+=1
				# write "Is Hit?"
				worksheet.write(row, cell, 0)
				cell+=1
				row+=1
		con.close()

main()
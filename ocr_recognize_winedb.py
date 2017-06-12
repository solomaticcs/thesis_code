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
import json
import time

def textSimilarity(s1, s2):
	longer = s1
	shorter = s2
	if len(s1) < len(s2):
		longer = s2
		shorter = s1
	longerLength = len(longer)
	if longerLength == 0:
		return 1.0
	return float(longerLength - levenshtein(longer, shorter)) / longerLength

def levenshtein(s1, s2):
	if len(s1) < len(s2):
		return levenshtein(s2, s1)

	# len(s1) >= len(s2)
	if len(s2) == 0:
		return len(s1)

	previous_row = range(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
			deletions = current_row[j] + 1       # than s2
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]

def ocr(filepath):
	filename=myfunc.path_leaf(filepath)

	# create directory
	ocr_directory = output_directory + "/" + filename;
	if not os.path.exists(ocr_directory):
		os.makedirs(ocr_directory)

	winelabel=myfunc.find_wine_label_v2(filepath,True)
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
	# width,height
	height, width = winelabelbinary.shape[:2]
	# filter contours
	filter_contours = myfunc.filter_contours(width,height,contours)

	text_region_image_path = []
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
		
		for k in range(0,len(ccl_bounding),1):
			# print "top:" + str(ccl_bounding[k]["top"]) + " bottom:" + str(ccl_bounding[k]["bottom"])
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
			bocrStrip = bocr.strip()
			bocrStripFilter = ''.join(e for e in bocrStrip if e.isalnum())
			if bocrStripFilter != "":
				ocr_recognized_text.append(bocrStripFilter)
				# add text region
				text_region_image_path.append(ocr_directory + "/text_region_"+str(k)+"_binary.png")
	else:
		print "no contour"

	return (text_region_image_path, ocr_recognized_text)

def main():
	if len(sys.argv)>1:
		global output_directory

		imgfolderpath=sys.argv[1]
		outputfilename=sys.argv[2]

		# create directory
		output_directory="./ocr_recognized_output/"
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)

		""" create xlsxwriter """
		# Create an new Excel file and add a worksheet.
		workbook = xlsxwriter.Workbook(output_directory + "/" + outputfilename + ".xlsx")
		worksheet = workbook.add_worksheet()
		titles=["Wine ID","Image Name", "Image Hit", "Cost Time"]
		row=0
		cell=0
		for title in titles:
			worksheet.write(row,cell,title)
			cell+=1
		worksheet2 = workbook.add_worksheet()
		titles2=["Image Name","Text Region Image","Recognize Text", "Hit Wine Name", "Is Hit?","Cost Time"]
		row2=0
		cell2=0
		for title2 in titles2:
			worksheet2.write(row2,cell2,title2)
			cell2+=1

		# sqlite
		con = sqlite3.connect("wine.sqlite")
		con.enable_load_extension(True)
		con.load_extension('./liblevenshtein.so')
		con.enable_load_extension(False)
		cur = con.cursor()

		sql = """SELECT `ID`,`Name`,`Rating`,`PriceTWD`,`Winery`,`Grapes`,
				`WineRegion`,`RegionalStyle`,`Country`,`FoodPairing`,`Review`,`ImageURL`,`URL` 
				FROM `wine-data-v2-db` """
		cur.execute(sql)
		alldatas = cur.fetchall()

		tmpinfo=[]
		tmpwineid=0
		tmpcnt=0
		row=1
		row2=1
		
		files = os.listdir(imgfolderpath)
		ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
		for filename in ordered_files:
			
			start_time = time.time()

			filepath = imgfolderpath+"/"+filename
			imagename = myfunc.path_name(filename)
			print imagename
			wineid=re.findall('\d+', imagename)[0]
			print wineid

			if tmpwineid != wineid:
				tmpwineid = wineid
				tmpinfo=[]
				tmpcnt=0

			text_region_image_path, ocr_recognized_text = ocr(filepath)

			image_is_hit = 0
			for k in range(0, len(ocr_recognized_text),1):
				words = []
				start_time2 = time.time()
				for i in range(0, len(alldatas), 1):
					word = {"id":alldatas[i][0],"name":"none", "value": 0}
					names =  alldatas[i][1].split(" ")
					for name in names:
						value = textSimilarity(name.lower(), ocr_recognized_text[k].lower())
						if value >= word["value"]:
							word["name"] = name
							word["value"] = value
					if word["value"] >= 0.7:
						words.append(word)
				end_time2 = time.time()
				cost_time2 = end_time2 - start_time2

				# for i in range(0, len(words), 1):
				# 	if words[i]["value"] >= 0.7:
				# 		filterWords.append(words[i])

				hit_wine_name_str = ""
				if len(words) > 0:
					image_is_hit = 1
					for i in range(0, len(words),1):
						words_record = json.dumps(words[i])
						print words_record
						hit_wine_name_str += words_record
					
				cell2 = 0
				# write "Image Name"
				worksheet2.write(row2, cell2, imagename)
				cell2+=1
				# write "Text Region Image"
				worksheet2.insert_image(row2, cell2, text_region_image_path[k])
				cell2+=1
				# write "Recognize Text"
				worksheet2.write(row2, cell2, ocr_recognized_text[k])
				cell2+=1
				# write "Hit Wine Name"
				worksheet2.write(row2, cell2, hit_wine_name_str)
				cell2+=1
				# write "is Hit?"
				if hit_wine_name_str != "":
					worksheet2.write(row2, cell2, 1)
				else:
					worksheet2.write(row2, cell2, 0)
				cell2+=1
				# write "Cost Time"
				worksheet2.write(row2, cell2, cost_time2)
				cell2+=1

				row2+=1
			
			end_time = time.time()
			cost_time = end_time - start_time

			tmpinfo.append({"wineid":wineid,"imagename":imagename,"image_is_hit":image_is_hit,"cost_time":cost_time})
			tmpcnt+=1

			if tmpcnt==5:
				for i in range(len(tmpinfo)):
					cell = 0
					# write "Wine ID"
					worksheet.write(row, cell,tmpinfo[i]["wineid"])
					cell+=1
					# write "Image Name"
					worksheet.write(row, cell, tmpinfo[i]["imagename"])
					cell+=1
					# write "Image Hit"
					worksheet.write(row, cell, tmpinfo[i]["image_is_hit"])
					cell+=1
					# write "Cost Time"
					worksheet.write(row, cell, tmpinfo[i]["cost_time"])
					cell+=1

					row+=1

		con.close()

main()
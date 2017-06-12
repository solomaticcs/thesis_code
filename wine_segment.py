import sys
import cv2
import myfunc

def main():
	if len(sys.argv)>1:
		filepath = sys.argv[1]
		winelabelv1 = myfunc.wine_segment_v1(filepath)
		winelabelv2 = myfunc.wine_segment_v2(filepath)
		winelabelv3 = myfunc.wine_segment_v3(filepath)

		# Show the image
		cv2.imshow('winelabelv1', winelabelv1)
		cv2.imshow('winelabelv2', winelabelv2)
		cv2.imshow('winelabelv3', winelabelv3)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

main()
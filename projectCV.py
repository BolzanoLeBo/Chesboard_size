import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from math import * 
from scipy.ndimage.filters import maximum_filter


#--------------------------NON USED FUNCTIONS-------------------------------------------

def corner_detect(input_img, origin_img) :
	#input img is an image in gray variation, and origin img a rgb one 
	img = np.copy(origin_img)
	corner = cv.cornerHarris(input_img,2,3,0.04)
	#result is dilated for marking the corners, not important
	dst = cv.dilate(corner,None)
	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst.max()]=[0,0,255]
	return img
	
def avarage_filter(source_img, size):
#size of the matrix edge 
	img = np.copy(source_img)
	#create filter matrix
	kernel = 1/(size**2) * np.ones((size,size))
	#using avarage filter 
	return cv.filter2D(img , -1, kernel)


def change_contr_bright(img, a, b): 
#simple contrast and britghtess change
	img2 = np.copy(img)
	for y in range (0, img.shape[0]) : 
		for x in range (0, img.shape[1]) : 
			if a*img2[y][x] + b < 255 : 
			#We have to check if we don't go out of the pixel value limit  
				img2[y][x] = ceil(a*img2[y][x] + b)
			else : 
				img2[y][x] = 255
	return img2


def linear_streching(gray_img, a_min, a_max) :

	img = np.copy(gray_img) 
	a_low = np.min(img) #the minimum value of pixel brightness 
	a_high = np.max(img) #the maximum value of pixel brightness 
	for y in range (0, img.shape[0]) : 
		for x in range (0, img.shape[1]) : 
			a = img[y][x]
			img[y][x] =a_min + (a - a_low)*(a_max-a_min)/(a_high- a_low)
	return img 



#-------------------------------------------------------------------------------------------



def gamma_correction(source_img, gamma) : 
	img = np.copy(source_img)
	img = (np.power((img/255), gamma)*255)
	img = img.astype(np.uint8)
	return img

def contour_rehaussement(source_img) : 
	img = np.copy(source_img)
	laplace = np.array([[0, 1, 0],
						[1, -4, 1], 
						[0, 1, 0]])
	img_filt = cv.filter2D(source_img , -1, laplace)

	return img + img_filt





def line_coord_change(r, theta) :
#change r theta cordonate to a (start, end) point
#We multiply by 1000 to simulate infinite line compared to the size of the image
	a = cos(theta)
	b = sin(theta)
	y0 = b*r
	x0 = a*r
	#we find to far away points in the line
	x1 = int(x0 - 1000*b)
	y1 = int(y0 + 1000*a)
	x2 = int(x0 + 1000*b)
	y2 = int(y0 - 1000*a)

	return [(x1,y1), (x2,y2)]



def lines_detector(source_img, img_edge, tresh) :
	
	img = np.copy(source_img)
	lines = cv.HoughLines(img_edge, 1, pi/180, tresh)
	
	for r_theta in lines : 
		arr = np.array(r_theta[0], dtype=np.float64)
		r, theta = arr
		line_coord = line_coord_change(r, theta)
		
		cv.line(img, line_coord[0], line_coord[1], (0,0,255), 1)

	return img



def lines_detector_P(source_img, img_edge, tresh, min_Length, max_Gap, line_tab) : 
	#Use a different dfonction for detecing lines 
	#WARNING modify the line_tab variable
	img = np.copy(source_img)
	lines = cv.HoughLinesP(
			img_edge, # Input edge image
			1, # Distance resolution in pixels
			np.pi/180, # Angle resolution in radians
			threshold=tresh, # Min number of votes for valid line
			minLineLength=min_Length, # Min allowed length of line
			maxLineGap=max_Gap # Max allowed gap between line for joining them
			)

	for point in lines : 
		x1,y1,x2,y2=point[0]
		cv.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
		line_tab.append(point[0])

	return img



def average_color(img):  
#get the average color of the image 
	return np.sum(img)/(np.shape(img)[0]*np.shape(img)[1]) 

def remove_in_spectrum(xcenter, ycenter, r, mag, ang) : 
	#remove a circle in the spectrum
	xshape = mag.shape[1]
	yshape = mag.shape[0]
	for y in range (ycenter - r, ycenter + r) : 
		for x in range (xcenter - r, xcenter + r) : 
			#if not outside of the image
			if (x >= 0 and x < xshape) and (y >= 0 and y < yshape) : 

				#if in the circle 
				if 2*(x - xcenter)**2 + 2*(y - ycenter)**2 <= r**2 : 
					mag[y][x] = 0


def remove_in_spectrum_out(xcenter, ycenter, r, mag, ang) : 
	#remove the frequency outside the circle 
	for y in range (0, mag.shape[0]) : 
		for x in range (0, mag.shape[1]) : 
			if 2*(x - xcenter)**2 + 2*(y - ycenter)**2 >= size**2 : 

				mag[y][x] = 0
				ang[y][x] = 0



def remove_periodic_noise(img): 
	
	#Do the fourier transform
	dft = np.fft.fft2(img)
	dft_shift = np.fft.fftshift(dft)

	mag = np.abs(dft_shift)
	ang = np.angle(dft_shift)


	#find local maxima 
	maxima = (mag == maximum_filter(mag,70))
	ymax, xmax = np.where(maxima)
	print(ymax, xmax)

	#center of the image
	xcenter = mag.shape[1]/2
	ycenter = mag.shape[0]/2


	#remove the frequency which do the noise
	for i in range (0, len(ymax)) : 
		x = xmax[i]
		y = ymax[i]
		if 2*(x - xcenter)**2 + 2*(y - ycenter)**2 >= 5**2 : 
			remove_in_spectrum(x, y, 25, mag, ang) 

	"""y1, x1, y2, x2 = ymax[0], xmax[0], ymax[2], xmax[2]
	t = 30
	remove_in_spectrum(x1, y1, t, mag)
	remove_in_spectrum(x2, y2, t, mag)"""


	#re-create the image
	combined = np.multiply(mag, np.exp(1j*ang))
	fftx = np.fft.ifftshift(combined)
	ffty = np.fft.ifft2(fftx)
	imgCombined = np.abs(ffty)

	f, axarr = plt.subplots(2,1) 

	axarr[0].imshow(20*np.log(mag), cmap = 'gray')
	axarr[1].imshow(imgCombined, cmap = 'gray')
	plt.show()


 # define a null callback function for Trackbar
def null(x):
     pass






def main(): 
	filename = "test.png"
	img = cv.imread(filename)
	#img = cv.resize(img, (250,250) )

	#creation of the trackbars to change parameters 
	cv.namedWindow('w')
	cv.createTrackbar("gamma", "w", 3, 10, null)
	cv.createTrackbar("treshold", "w", 100, 255, null)
	cv.createTrackbar("tresholdP", "w", 100, 255, null)
	cv.createTrackbar("cannyMin", "w", 100, 255, null)
	cv.createTrackbar("cannyMax", "w", 150, 255, null)
	cv.createTrackbar("houghmin", "w", 75, 255, null)
	cv.createTrackbar("houghmax", "w", 50, 255, null)

	loop = True 
	init = True 
	while loop : 
		key = cv.waitKey(1) & 0xFF

		#press 0 to close 
		if key == ord('0'):
			loop = False 
			cv.destroyAllWindows()

		#press L to load the changes in the parameters 
		if key == ord('l') or init:
			init = False 
			g = cv.getTrackbarPos('gamma','w')/10
			tresh = cv.getTrackbarPos("treshold", "w")
			treshP = cv.getTrackbarPos("tresholdP", "w")
			cmin = cv.getTrackbarPos("cannyMin", "w")
			cmax = cv.getTrackbarPos("cannyMax", "w")
			hmin = cv.getTrackbarPos("houghmin", "w")
			hmax =cv.getTrackbarPos("houghmax", "w")

			
#---------------------IMAGE PROCESSING-----------------------------



			final_img = np.copy(img)

			#create image with only gray variation 
			gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

			
			#remove noise
			img_filt = cv.GaussianBlur(gray,(3,3),0)

			#change contrast and brightness 
			img_contr = gamma_correction(img_filt, g) #0.3	

			contour = contour_rehaussement(img_contr)

			#only draw the edges 
			img_edge = cv.Canny(contour, cmin, cmax, apertureSize=3) #100 125
			

			line_tab = []

			cv.imshow('img', img)

			cv.imshow('filt',img_filt)
			cv.imshow('contrast', img_contr)
			cv.imshow('contour', contour)

			cv.imshow('edges',img_edge)
			
			cv.imshow('lines', lines_detector(final_img, img_edge, tresh)) #100
			cv.imshow('linesP', lines_detector_P(final_img, img_edge, hmin, hmax, treshP, line_tab)) #75 50 100
			





main()
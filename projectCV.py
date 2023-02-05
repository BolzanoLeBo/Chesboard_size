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

	return [x1,y1, x2,y2]



def lines_detector(img_edge, tresh) :
	
	line_tab = []
	line_tab.append([1,1,0,0])

	lines = cv.HoughLines(img_edge, 1, pi/180, tresh)
	
	for r_theta in lines : 
		arr = np.array(r_theta[0], dtype=np.float64)
		r, theta = arr
		line_coord = line_coord_change(r, theta)
		copy = False
		for line in line_tab :
			if lines_close(line_coord, line) : 
				copy = True 
		if not copy:			
			line_tab.append(line_coord)

	return line_tab



def lines_detector_P(img_edge, tresh, min_Length, max_Gap) : 
	#Use a different dfonction for detecing lines 
	
	line_tab = []
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
		copy = False
		for line in line_tab : 
			if lines_close([x1,y1,x2,y2], line) : 
				copy = True 
		if not copy:
			line_tab.append([x1, y1, x2, y2])
	return line_tab		



def average_color(img):  
#get the average color of the image 
	return np.sum(img)/(np.shape(img)[0]*np.shape(img)[1]) 

def remove_in_spectrum(xcenter, ycenter, r, mag) : 
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

def remove_sinusoidal_noise(img):
	# Convert the image to a 2D numpy array
	img = np.array(img)

	# Apply the FFT to the image
	f = np.fft.fft2(img)

	# Shift the zero-frequency component to the center
	fshift = np.fft.fftshift(f)

	#To see if components were removed
	removed = False 

	# Get the rows and columns of the image
	rows, cols = img.shape

	 # Set the noise-removal threshold for the other frequency components
	other_freq_threshold = 1000 * average_color(np.abs(fshift))


	 # Zero out the high-frequency components that correspond to noise
	for y in range(rows):
		for x in range(cols):
			if not (y == rows//2 and x == cols//2):
				if np.abs(fshift[y][x]) > other_freq_threshold:
					fshift[y][x] = 0
					removed = True

	# Shift the zero-frequency component back to the original location
	f_ishift = np.fft.ifftshift(fshift)

	# Apply the inverse FFT to the filtered image
	img_back = np.fft.ifft2(f_ishift)

	# Convert the result back to an 8-bit unsigned integer
	img_back = np.abs(img_back).astype(np.uint8)
	return [img_back, removed]


def line_equation(line_points) : 
	x1, y1, x2, y2 = line_points
	if x1 == x2 :  #if it is a vertical line 
		x1 += 0.0000001

	m = (y1 - y2) / (x1 - x2)
	b = y1 - m* x1
	return [m, b]

def in_image(x, y, img ) : 
	lx = img.shape[1]
	ly = img.shape[0]

	return (x < lx and x > 0) and (y < ly and y > 0)
 
def find_all_intersection(img, line_tab) : 
	tab_intersect = []
	for i in range (0, len(line_tab)) :
		#we will find intersection between this line and all the others  
		m1, b1 = line_equation(line_tab[i])
		for j in range (i, len(line_tab)) : 
			#equation for others lines
			m2, b2 = line_equation(line_tab[j])
			if (m1 - m2) != 0 : 
				x = (b2 - b1) / (m1 - m2)
				y = m1 * x + b1

				if in_image(x, y, img) : 
					copy = False
					for point in tab_intersect : 
						if point_close([x, y], point) : 
							copy = True 
					if not copy:
						tab_intersect.append([x, y])
	return tab_intersect



 # define a null callback function for Trackbar
def null(x):
	 pass

def point_close (p1, p2 ): 
	treshold = 20
	return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < treshold

def lines_close(l1, l2) :
	
	treshold1 = 10 #the distance between the two interesection with the y axis 
	treshold2 = 35 #difference between the 

	#equation for line 1
	m1, b1 = line_equation(l1)

	#equation for line 2
	m2, b2 = line_equation(l2)

	return abs(b1 - b2) < treshold1 and abs(m1 - m2) < treshold2

def draw_lines(source_img, line_tab) : 
	img = np.copy(source_img)
	for line in line_tab : 
		x1, y1, x2, y2 = line
		cv.line(img, (x1, y1), (x2, y2), (0,0,255), 1)
	return img



def main(): 
	filename = "test.png"
	img = cv.imread(filename)
	#img = cv.resize(img, (250,250) )

	final_img = np.copy(img)

	#create image with only gray variation 
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	img_nosin, sin_removed = remove_sinusoidal_noise(gray)

	#creation of the trackbars to change parameters 
	cv.namedWindow('w')
	

	if not sin_removed : 
		cv.createTrackbar("gamma", "w", 3, 10, null)
		cv.createTrackbar("cannyMin", "w", 50, 100, null)
		cv.createTrackbar("cannyMax", "w", 170, 200, null)
	else : 
		cv.createTrackbar("gamma", "w", 15, 50, null)
		cv.createTrackbar("cannyMin", "w", 10, 100, null)
		cv.createTrackbar("cannyMax", "w", 50, 200, null)

	cv.createTrackbar("treshold", "w", 100, 255, null)
	cv.createTrackbar("tresholdP", "w", 100, 150, null)
	cv.createTrackbar("houghmin", "w", 50, 100, null)
	cv.createTrackbar("houghmax", "w", 10, 50, null)

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




			
			#remove noise
			if sin_removed : 
				img_filt = cv.GaussianBlur(img_nosin,(3,3),0)
			else :
				img_filt = cv.GaussianBlur(gray,(3,3),0)

			#change contrast and brightness 

			img_contr = gamma_correction(img_filt, g)

			contour = contour_rehaussement(img_contr)

			#only draw the edges 
			img_edge = cv.Canny(contour, cmin, cmax, apertureSize=3)

			line_tab = []

			cv.imshow('img', img)

			cv.imshow('filt',img_filt)
			cv.imshow('contrast', img_contr)
			cv.imshow('contour', contour)

			cv.imshow('edges',img_edge)
			
			
			line_tabP = lines_detector_P(img_edge, hmin, hmax, treshP)
			line_tab = lines_detector(img_edge, tresh)

			cv.imshow('lines', draw_lines(final_img, line_tab))
			cv.imshow('linesP', draw_lines(final_img, line_tabP)) 

			intersectP = find_all_intersection(final_img, line_tabP)
			img2 = np.copy(final_img)
			#
			for point in intersectP : 
				cv.circle(img2, (int(point[0]), int(point[1])), 5, (0,255,0), 3)
			cv.imshow('img2', img2) 
			print(len(intersectP), len(line_tab))

			
		





main()
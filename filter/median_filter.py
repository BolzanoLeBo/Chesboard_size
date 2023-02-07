import cv2 as cv
import numpy as np

# Load the image
filename = "../Chessboard_0481_0.png"
img = cv.imread(filename)

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur with a kernel size of (15,15)
blurred = cv.GaussianBlur(gray, (15, 15), 0)

# Apply the median filter
median = cv.medianBlur(gray, 5)

# Apply thresholding
_, thresh = cv.threshold(gray, 25, 255, cv.THRESH_BINARY)

# Create a sharpening filter
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened = cv.filter2D(blurred, -1, kernel)

# Convert the image to the correct data type
gamma = np.uint8(np.power(blurred / 255, 1.9 / 2.2) * 255)

# Apply the Canny edge detection algorithm
edges = cv.Canny(blurred, 10, 30)
edges2 = cv.Canny(sharpened, 40, 100)
edges_median = cv.Canny(median, 50, 120)

# Show the original and blurred images
# cv.imshow("Original", img)
# cv.imshow("gray", gray)
# cv.imshow("Blurred", blurred)
cv.imshow("Median", median)
# cv.imshow("Thresholded", thresh)
# cv.imshow("Sharpened", sharpened)
# cv.imshow("Gamma Corrected", gamma)
# cv.imshow("Edges", edges)
# cv.imshow("Edges sharpened", edges2)
cv.imshow("Edges median", edges_median)


# Create a black image with the same size as the input image
overlay = np.zeros(img.shape, dtype=np.uint8)

# Copy the edges image to the green channel of the overlay image
overlay[..., 0] = edges_median
overlay[..., 1] = edges_median
overlay[..., 2] = edges_median

# Overlay the edges on the original image
result = cv.addWeighted(img, 0.7, overlay, 0.3, 0)
cv.imshow("Overlay", result)


# Wait for key press and close windows
cv.waitKey(0)
cv.destroyAllWindows()

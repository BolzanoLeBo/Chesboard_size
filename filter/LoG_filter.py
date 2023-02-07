import cv2
import numpy as np

# Load the image
img = cv2.imread("../Chessboard_00451.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian filter
gaussian = cv2.GaussianBlur(gray, (15, 15), 0)

# Apply Laplacian of Gaussian filter
log = cv2.Laplacian(gaussian, cv2.CV_8U, ksize=5)

# Show the original, Gaussian, and Laplacian of Gaussian images
cv2.imshow("Original", img)
cv2.imshow("Gaussian", gaussian)
cv2.imshow("Laplacian of Gaussian", log)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()

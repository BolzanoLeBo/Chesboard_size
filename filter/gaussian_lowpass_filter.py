import cv2 as cv

# Load the image
img = cv.imread("../Chessboard_0481_0.jpg")

# Apply Gaussian blur with a kernel size of (15,15)
blurred = cv.GaussianBlur(img, (15, 15), 0)

# Show the original and blurred images
cv.imshow("Original", img)
cv.imshow("Blurred", blurred)

# Wait for key press and close windows
cv.waitKey(0)
cv.destroyAllWindows()

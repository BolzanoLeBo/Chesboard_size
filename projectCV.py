import numpy as np
import cv2 as cv
import line_and_point_detect as lp


# --------------------------NON USED FUNCTIONS-------------------------------------------


# def corner_detect(input_img, origin_img):
#     # input img is an image in gray variation, and origin img a rgb one
#     img = np.copy(origin_img)
#     corner = cv.cornerHarris(input_img, 2, 3, 0.04)
#     # result is dilated for marking the corners, not important
#     dst = cv.dilate(corner, None)
#     # Threshold for an optimal value, it may vary depending on the image.
#     img[dst > 0.01 * dst.max()] = [0, 0, 255]
#     return img


# def avarage_filter(source_img, size):
#     # size of the matrix edge
#     img = np.copy(source_img)
#     # create filter matrix
#     kernel = 1 / (size**2) * np.ones((size, size))
#     # using avarage filter
#     return cv.filter2D(img, -1, kernel)


# def change_contr_bright(img, a, b):
#     # simple contrast and britghtess change
#     img2 = np.copy(img)
#     for y in range(0, img.shape[0]):
#         for x in range(0, img.shape[1]):
#             if a * img2[y][x] + b < 255:
#                 # We have to check if we don't go out of the pixel value limit
#                 img2[y][x] = ceil(a * img2[y][x] + b)
#             else:
#                 img2[y][x] = 255
#     return img2


# def linear_streching(gray_img, a_min, a_max):
#     img = np.copy(gray_img)
#     a_low = np.min(img)  # the minimum value of pixel brightness
#     a_high = np.max(img)  # the maximum value of pixel brightness
#     for y in range(0, img.shape[0]):
#         for x in range(0, img.shape[1]):
#             a = img[y][x]
#             img[y][x] = a_min + (a - a_low) * (a_max - a_min) / (a_high - a_low)
#     return img


# -------------------------------------------------------------------------------------------


def gamma_correction(source_img, gamma):
    img = np.copy(source_img)
    """img = (np.power((img/255), gamma)*255)
	img = img.astype(np.uint8)
	return img"""
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
    return cv.LUT(img, lookUpTable)


def automatic_gamma_correction(img):
    # Averaga and variance of the image
    mean, stddev = cv.meanStdDev(img)
    mean = mean[0][0] / 256
    variance = stddev[0][0] / 256
    print(variance)
    k = 0.5
    gamma = np.log(mean + k * variance) / np.log(0.5)

    # gamma corrextion
    return gamma_correction(img, gamma)


def contour_rehaussement(source_img):
    img = np.copy(source_img)
    laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img_filt = cv.filter2D(source_img, -1, laplace)

    return img + img_filt


def average_color(img):
    # get the average color of the image
    return np.sum(img) / (np.shape(img)[0] * np.shape(img)[1])


def remove_in_spectrum(xcenter, ycenter, r, mag):
    # remove a circle in the spectrum
    xshape = mag.shape[1]
    yshape = mag.shape[0]
    for y in range(ycenter - r, ycenter + r):
        for x in range(xcenter - r, xcenter + r):
            # if not outside of the image
            if (x >= 0 and x < xshape) and (y >= 0 and y < yshape):
                # if in the circle
                if 2 * (x - xcenter) ** 2 + 2 * (y - ycenter) ** 2 <= r**2:
                    mag[y][x] = 0


def remove_in_spectrum_out(xcenter, ycenter, r, mag, ang):
    # remove the frequency outside the circle
    for y in range(0, mag.shape[0]):
        for x in range(0, mag.shape[1]):
            if 2 * (x - xcenter) ** 2 + 2 * (y - ycenter) ** 2 >= size**2:
                mag[y][x] = 0
                ang[y][x] = 0


def remove_sinusoidal_noise(img):
    # Convert the image to a 2D numpy array
    img = np.array(img)

    # Apply the FFT to the image
    f = np.fft.fft2(img)

    # Shift the zero-frequency component to the center
    fshift = np.fft.fftshift(f)

    # To see if components were removed
    removed = False

    # Get the rows and columns of the image
    rows, cols = img.shape

    # Set the noise-removal threshold for the other frequency components
    other_freq_threshold = 1000 * average_color(np.abs(fshift))

    # Zero out the high-frequency components that correspond to noise
    for y in range(rows):
        for x in range(cols):
            if not (y == rows // 2 and x == cols // 2):
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


def null(x):
    pass


def main():
    # filename = "Chessboard_00451_2.png" # sin
    filename = "Chessboard_00451.png"  # normal
    # filename = "Chessboard_0481_0.png" # noise
    # filename = "Chessboard_0481.png"
    # filename = "Chessboard_0511.png"
    # filename = "Chessboard_0541.png"
    # filename = "Chessboard_00601.png"
    # filename = "Chessboard0631.png"

    img = cv.imread(filename)

    final_img = np.copy(img)

    # create image with only gray variation
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_nosin, sin_removed = remove_sinusoidal_noise(gray)

    # creation of the trackbars to change parameters
    cv.namedWindow("w")

    if not sin_removed:
        cv.createTrackbar("gamma", "w", 3, 10, null)
        cv.createTrackbar("cannyMin", "w", 50, 100, null)
        cv.createTrackbar("cannyMax", "w", 170, 200, null)
    else:
        cv.createTrackbar("gamma", "w", 15, 50, null)
        cv.createTrackbar("cannyMin", "w", 10, 100, null)
        cv.createTrackbar("cannyMax", "w", 50, 200, null)

    cv.createTrackbar("treshold", "w", 100, 255, null)
    cv.createTrackbar("tresholdP", "w", 100, 150, null)
    cv.createTrackbar("houghmin", "w", 50, 100, null)
    cv.createTrackbar("houghmax", "w", 10, 50, null)

    loop = True
    init = True
    while loop:
        key = cv.waitKey(1) & 0xFF

        # press 0 to close
        if key == ord("0"):
            loop = False
            cv.destroyAllWindows()

        # press L to load the changes in the parameters
        if key == ord("l") or init:
            init = False

            g = cv.getTrackbarPos("gamma", "w") / 10
            tresh = cv.getTrackbarPos("treshold", "w")
            treshP = cv.getTrackbarPos("tresholdP", "w")
            cmin = cv.getTrackbarPos("cannyMin", "w")
            cmax = cv.getTrackbarPos("cannyMax", "w")
            hmin = cv.getTrackbarPos("houghmin", "w")
            hmax = cv.getTrackbarPos("houghmax", "w")

            # ---------------------IMAGE PROCESSING-----------------------------

            # remove noise
            if sin_removed:
                img_filt = cv.GaussianBlur(img_nosin, (3, 3), 0)
            else:
                img_filt = cv.GaussianBlur(gray, (3, 3), 0)

            # change contrast and brightness

            img_contr = automatic_gamma_correction(img_filt)

            contour = contour_rehaussement(img_contr)

            # only draw the edges
            img_edge = cv.Canny(contour, cmin, cmax, apertureSize=3)

            line_tab = []

            cv.imshow("img", img)

            cv.imshow("filt", img_filt)
            cv.imshow("contrast", img_contr)
            cv.imshow("contour", contour)

            cv.imshow("edges", img_edge)

            line_tabP = lp.lines_detector_P(img_edge, hmin, hmax, treshP)
            line_tab = lp.lines_detector(img_edge, tresh)

            cv.imshow("lines", lp.draw_lines(final_img, line_tab))
            cv.imshow("linesP", lp.draw_lines(final_img, line_tabP))

            intersectP = lp.find_all_intersection(final_img, line_tabP)
            img2 = np.copy(final_img)
            #
            for point in intersectP:
                cv.circle(img2, (int(point[0]), int(point[1])), 5, (0, 255, 0), 3)
            cv.imshow("img2", img2)
            print(len(intersectP), len(line_tab))


main()

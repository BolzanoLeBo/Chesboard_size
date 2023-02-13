import numpy as np
import cv2 as cv
from math import *


def line_equation(line_points):
    x1, y1, x2, y2 = line_points
    if x1 == x2:  # if it is a vertical line
        x1 += 0.0000001

    m = (y1 - y2) / (x1 - x2)
    b = y1 - m * x1
    return [m, b]


def line_equation2(line_points):
    # we do the equation as we invert x and y, it is useful to compare vertical lines
    y1, x1, y2, x2 = line_points
    if x1 == x2:  # if it is a vertical line
        x1 += 0.0000001

    m = (y1 - y2) / (x1 - x2)
    b = y1 - m * x1
    return [m, b]


def in_image(x, y, img):
    lx = img.shape[1]
    ly = img.shape[0]

    return (x < lx and x > 0) and (y < ly and y > 0)


def find_all_intersection(img, line_tab):
    tab_intersect = []

    # also count the number of points in the first line
    nb_point_side = 0

    for i in range(0, len(line_tab)):
        # we will find intersection between this line and all the others
        m1, b1 = line_equation(line_tab[i])
        for j in range(i, len(line_tab)):
            # equation for others lines
            m2, b2 = line_equation(line_tab[j])
            if (m1 - m2) != 0:
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1

                if in_image(x, y, img):
                    copy = False
                    for point in tab_intersect:
                        if point_close([x, y], point):
                            copy = True
                    if not copy:
                        tab_intersect.append([x, y])
                        if i == 0:
                            nb_point_side += 1
    return tab_intersect, nb_point_side

    # define a null callback function for Trackbar


def point_close(p1, p2):
    treshold = 20
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < treshold


def lines_close(l1, l2):
    treshold1 = 10  # the distance between the two interesection with the y axis
    treshold2 = 35  # difference between the

    # equation for line 1
    m1, b1 = line_equation(l1)

    # equation for line 2
    m2, b2 = line_equation(l2)
    if m1 > 1 and m2 > 1:
        # it is vertical line so comparaing b is no-sense
        # because b is the intersection with y axis so if line are almost vertical there will be huge difference between there b
        # we have to change the sense of the equation
        # equation for line 1
        m1, b1 = line_equation2(l1)

        # equation for line 2
        m2, b2 = line_equation2(l2)
        return abs(b1 - b2) < treshold1 and abs(m1 - m2) < treshold2
    else:
        return abs(b1 - b2) < treshold1 and abs(m1 - m2) < treshold2

def line_norm(line) :
    x1, y1, x2, y2 = line 
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def draw_lines(source_img, line_tab):
    img = np.copy(source_img)
    for line in line_tab:
        x1, y1, x2, y2 = line
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return img


def line_coord_change(r, theta):
    # change r theta cordonate to a (start, end) point
    # We multiply by 1000 to simulate infinite line compared to the size of the image
    a = cos(theta)
    b = sin(theta)
    y0 = b * r
    x0 = a * r
    # we find to far away points in the line
    x1 = int(x0 - 1000 * b)
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 + 1000 * b)
    y2 = int(y0 - 1000 * a)

    return [x1, y1, x2, y2]


def lines_detector(img_edge, tresh):
    line_tab = []
    line_tab.append([1, 1, 0, 0])

    lines = cv.HoughLines(img_edge, 1, pi / 180, tresh)

    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        line_coord = line_coord_change(r, theta)
        copy = False
        for line in line_tab:
            if lines_close(line_coord, line):
                copy = True
        if not copy:
            line_tab.append(line_coord)

    return line_tab


def lines_detector_P(img_edge, tresh, min_Length, max_Gap):
    # Use a different dfonction for detecing lines

    line_tab = []

    lines = cv.HoughLinesP(
        img_edge,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=tresh,  # Min number of votes for valid line
        minLineLength=min_Length,  # Min allowed length of line
        maxLineGap=max_Gap,  # Max allowed gap between line for joining them
    )

    for point in lines:
        x1, y1, x2, y2 = point[0]
        copy = False
        #line = point[0]
        for line in line_tab:
            if lines_close([x1, y1, x2, y2], line):
                copy = True
                if copy and line_norm([x1,y1,x2,y2]) >= line_norm(line) : 
                    #we want to keep the longest o,e 
                    print(line_tab)
                    line_tab.remove(line)
                    print(line_tab)
                    copy = False

        if not copy :
            line_tab.append([x1, y1, x2, y2])
    return line_tab

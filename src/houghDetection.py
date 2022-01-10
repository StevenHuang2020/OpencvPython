import cv2
import numpy as np
from mainImagePlot import plotImagList
from ImageBase import loadImg

# reference
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
# https://en.wikipedia.org/wiki/Hough_transform
# https://en.wikipedia.org/wiki/Canny_edge_detector


def houghLineDtection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    newImg = img.copy()
    if 0:
        minLineLength = 800
        maxLineGap = 20
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength, maxLineGap)
        if lines is not None:
            print('lines=', len(lines[0]))
            for x1, y1, x2, y2 in lines[0]:
                print(x1, x2)
                print(y1, y2)
                cv2.line(newImg, (x1, y1), (x2, y2), (255, 0, 0), 2)
    else:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is not None:
            print('lines=', len(lines[0]))
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(newImg, (x1, y1), (x2, y2), (0, 255, 0), 2)

    ls, nameList = [], []
    ls.append(img), nameList.append('Original')
    ls.append(gray), nameList.append('gray')
    ls.append(edges), nameList.append('edges')
    ls.append(newImg), nameList.append('newImg')
    plotImagList(ls, nameList)


def HoughCircleDetect(img):
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print('img:', img.shape, img.dtype)
    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    ls, nameList = [], []
    ls.append(img), nameList.append('Original')
    ls.append(cimg), nameList.append('cimg')
    plotImagList(ls, nameList)


if __name__ == "__main__":
    # img = loadImg(r'./res/sudoku-original.jpg') #Road_in_Norway.jpg
    # houghLineDtection(img)

    img = loadImg(r'./res/opencv-logo-white.png')
    HoughCircleDetect(img)

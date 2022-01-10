import cv2
import numpy as np
from mainImagePlot import plotImagList
from ImageBase import loadImg


def gradientImg(img):
    # img = grayImg(img)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = laplacian.astype(np.uint8)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobelx = sobelx.astype(np.uint8)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobely = sobely.astype(np.uint8)

    ls, nameList = [], []
    ls.append(img), nameList.append('Original')
    ls.append(laplacian), nameList.append('laplacian')
    ls.append(sobelx), nameList.append('sobelx')
    ls.append(sobely), nameList.append('sobely')
    plotImagList(ls, nameList, title='Fliter Image', gray=False)


if __name__ == "__main__":
    img = loadImg(r'./res/Lenna.png')  # Road_in_Norway.jpg
    gradientImg(img)

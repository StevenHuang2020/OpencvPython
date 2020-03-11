#python3
#Steven matplot hisgram of image

import numpy as np
import cv2.cv2 as cv2
from matplotlib import pyplot as plt

def plotHist(file,mode=cv2.IMREAD_COLOR):
    img = cv2.imread(file,mode)
    return plotHistImg(img)
    
def getImagChannel(img):
    if img.ndim == 3: #color r g b channel
        return 3
    return 1  #only one channel

def plotHistImg(img):
    color = ('b','g','r')
    chn = getImagChannel(img)
    for i in range(chn):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = color[i])
        plt.xlim([0,256])
    plt.show()

def getHist(file,mode=cv2.IMREAD_COLOR):
    img = cv2.imread(file,mode)
    return getHistImg(img)

def getHistImg(img):
    chn = getImagChannel(img)
    hists = []
    for i in range(chn):
        hists.append(cv2.calcHist([img],[i],None,[256],[0,256]))
    return hists

def plotHistGray(file):
    img = cv2.imread(file,0)
    return plotHistGrayImg(img)

def plotHistGrayImg(img):
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

def plotColorHist(file):
    img = cv2.imread(file)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def main():
    file=r'./res/Lenna.png'
    #plotHistGray(file)
    #plotColorHist(file)
    plotHist(file,cv2.IMREAD_GRAYSCALE)
    pass

if __name__=='__main__':
    main()

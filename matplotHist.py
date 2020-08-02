#python3
#Steven matplot hisgram of image

import numpy as np
import cv2
from matplotlib import pyplot as plt
from ImageBase import *


def plotHistImg(img):
    color = ('b','g','r')
    chn = getImagChannel(img)
    print(chn)
    for i in range(chn):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        #print(type(histr),histr.shape)
        #print(histr)
        #print(histr.ravel())
        plt.plot(histr,color = color[i])
        #plt.hist(histr.ravel(), 255, facecolor='blue', alpha=0.5)
        plt.xlim([0,256])
    plt.show()
  
def getImgHist(img):
    chn = getImagChannel(img)
    hists = []
    for i in range(chn):
        hists.append(cv2.calcHist([img],[i],None,[256],[0,256]))
    return hists

def getImgHist256Img(img):
    chn = getImagChannel(img)
    hist256Imgs = []
    for i in range(chn):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        hist256Imgs.append(getHist256ImgFromHist(hist))
    return hist256Imgs

def getHist256ImgFromHist(hist,color=[255,255,255]):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8) #[256,256,3] [256,256]
    hpt = int(0.9* 256)

    for h in range(256):
        #intensity = int(hist[h]*hpt/maxVal)
        intensity = hist[h]*hpt/maxVal
        #print(h,intensity)
        cv2.line(histImg,(h,256), (h,256-intensity), color)
    return histImg

def plotHistGrayImg(img):
    img = grayImg(img)
    plt.hist(img.ravel(),256,[0,256])
    plt.show()


def main():
    file=r'./res/Lenna.png'
    img = loadImg(file)
    #plotHistGrayImg(img)
    plotHistImg(img)
    pass

if __name__=='__main__':
    main()

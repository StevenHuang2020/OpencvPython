#python3
import numpy as np
from ImageBase import *
from mainImagePlot import plotImagList
from videoCap import *
from matplotlib import pyplot as plt
import cv2

def capColor():
    cap = openCap()
    while(1):
        # Take each frame
        _, frame = cap.read()
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    
def testFlip(img):
    gray = grayImg(img)
    color = colorSpace(img)
    #capColor()
    #thresHold(img)
    #thresHold2(img)
    flip = flipImg(img)
    flip2 = flipImg(img,False)
    grayFlip = flipImg(gray)
    
    ls,nameList = [],[]
    #ls.append(img),nameList.append('Original')
    #ls.append(color),nameList.append('color')
    ls.append(gray),nameList.append('gray')
    ls.append(grayFlip),nameList.append('grayFlip')
    
    ls.append(flip),nameList.append('flip')
    ls.append(flip2),nameList.append('flip2')
   
    plotImagList(ls,nameList,gray=False)
    
def testGrayImg(img):
    gray = grayImg(img)
    grayMean = custGray(img)
    grayC = custGray(img,False)
    #print(grayC[:1,:5])
    #infoImg(grayMean)
    
    ls,nameList = [],[]
    #ls.append(img),nameList.append('Original')
    ls.append(gray),nameList.append('gray')
    ls.append(grayMean),nameList.append('grayMean')
    ls.append(grayC),nameList.append('grayC')
    
    plotImagList(ls,nameList,gray=True)
    
def testDifferImg(img):
    difRL = differImg(img)
    difUD = differImg(img,False)
    gray = grayImg(img)
    difGrayRL = differImg(gray)
    difGrayUD = differImg(gray,False)
    
    ls,nameList = [],[]
    ls.append(difRL),nameList.append('difRL')
    ls.append(difUD),nameList.append('difUD')
    ls.append(difGrayRL),nameList.append('difGrayRL')
    ls.append(difGrayUD),nameList.append('difGrayUD')
    plotImagList(ls, nameList, gray=True)
    
def testDiffer2Img(img):
    img = grayImg(img)
    difRL = differImg(img)
    difUD = differImg(img,False)
   
    difRL2 = differImg(difRL)
    difUD2 = differImg(difUD,False)
    
    ls,nameList = [],[]
    ls.append(difRL),nameList.append('difRL')
    ls.append(difUD),nameList.append('difUD')
    ls.append(difRL2),nameList.append('difRL2')
    ls.append(difUD2),nameList.append('difUD2')
    plotImagList(ls, nameList, gray=True)

def testPyramid(img):
    #img = grayImg(img)
    pyramid = pyramidImg(img)
    pyramid2 = pyramidImg(pyramid)
    pyramid4 = pyramidImg(pyramid2)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('img')
    ls.append(pyramid),nameList.append('pyramid')
    ls.append(pyramid2),nameList.append('pyramid2')
    ls.append(pyramid4),nameList.append('pyramid4')
    plotImagList(ls, nameList)
    
if __name__ == "__main__":
    img = loadImg(r'./res/Lenna.png')
    #img = loadImg(r'./res/shudu2.jpg',0)
    #infoImg(img)
    
    #testFlip(img)
    #testGrayImg(img)
    testDifferImg(img)
    #testDiffer2Img(img)
    #print('mean,deviation=',meanImg(img),deviationImg(img))
    #testPyramid(img)
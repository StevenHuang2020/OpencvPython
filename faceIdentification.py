#python3
#Steven face identification
#process: 1.preprecess 2. face position 3.feature extraction 4.face recognition
import os,sys
import cv2.cv2 as cv2
import numpy as np

from CascadeClassifier import CascadeClassifier
from ImageBase import*
from mainImagePlot import plotImagList

def main():
    cv2.useOptimized()
    file = r'./res/Lenna.png' # r'./res/face.jpg'#
    
    print('Number of parameter:', len(sys.argv))
    print('Parameters:', str(sys.argv))
    if len(sys.argv)>1:
        file = sys.argv[1]

    cascPath=r'./res/haarcascade_frontalface_default.xml'

    img = loadImg(file,mode=cv2.IMREAD_COLOR) # IMREAD_GRAYSCALE IMREAD_COLOR
    img = changeBgr2Rbg(img)
    
    faceROI = CascadeClassifier(cascPath)
 
    faceR=faceROI.getDetectImg(img)
    face = faceROI.detecvFaceImgOne(img)
    faceGray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    ls = []
    names=[]
    ls.append(img),names.append('Original')
    ls.append(faceR),names.append('faceR')
    ls.append(face),names.append('face')
    ls.append(faceGray),names.append('faceGray')

    plotImagList(ls,names)

    #writeImg(face,'./res/myface_.png')
    #writeImg(faceGray,'./res/myface_gray.png')

if __name__=="__main__":
    main()


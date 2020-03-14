#python3
#Steven face identification
#process: 1.preprecess 2. face position 3.feature extraction 4.face recognition
import os,sys
import cv2.cv2 as cv2
import numpy as np

from CascadeClassifier import CascadeClassifier
from imageBase import ImageBase
from mainImagePlot import plotImagList

def main():
    cv2.useOptimized()
    file = r'./res/Lenna.png' # r'./res/face.jpg'#
    
    print('Number of parameter:', len(sys.argv))
    print('Parameters:', str(sys.argv))
    if len(sys.argv)>1:
        file = sys.argv[1]

    cascPath=r'./res/haarcascade_frontalface_default.xml'

    img = ImageBase(file,mode=cv2.IMREAD_COLOR) # IMREAD_GRAYSCALE IMREAD_COLOR

    #print(img.infoImg())
    faceROI = CascadeClassifier(cascPath)
 
    faceR=faceROI.getDetectImg(img.image)
    face = faceROI.detecvFaceImgOne(img.image)
    faceGray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    ls = []
    ls.append(img.image)
    ls.append(faceR)
    ls.append(face)
    ls.append(faceGray)

    plotImagList(ls)

    #img.writeImg(face,'./res/myface_.png')
    #img.writeImg(faceGray,'./res/myface_gray.png')

if __name__=="__main__":
    main()


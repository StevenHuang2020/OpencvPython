#python3
#Steven image convolution modoule

import cv2.cv2 as cv2 #pip install opencv-python
import matplotlib.pyplot as plt
from ImageBase import *
from mainImagePlot import plotImagList

def main():
    file = './res/Lenna.png' #r'./res/obama.jpg'#
    img = loadImg(file,mode=cv2.IMREAD_COLOR) # IMREAD_GRAYSCALE IMREAD_COLOR
    img = changeBgr2Rbg(img)
    img = grayImg(img)

    kernel = np.ones((3,3), np.float32)/9
    #print(kernel)
    k_Robert1 = np.array([[1,0],
                        [0,-1]])

    k_Robert2 = np.array([[0,1],
                        [-1,0]])

    k_Sobel1 = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])

    k_Sobel2 = k_Sobel1.T
    k_Laplance = np.array([[0,-1,0],
                        [-1,4,-1],
                        [0,-1,0]])

    k_Prewitt1 = np.array([[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]])
    k_Prewitt2 = k_Prewitt1.T

    conImg = convolutionImg(img,kernel)
    k_Robert1Img = convolutionImg(img,k_Robert1)
    k_Robert2Img = convolutionImg(img,k_Robert2)

    k_Sobel1Img = convolutionImg(img,k_Sobel1)
    k_Sobel2Img = convolutionImg(img,k_Sobel2)
    k_LaplanceImg = convolutionImg(img,k_Laplance)

    k_Prewitt1Img = convolutionImg(img,k_Prewitt1)
    k_Prewitt2Img = convolutionImg(img,k_Prewitt2)

    imgList = []
    nameList = []
    imgList.append(img), nameList.append('Original')
    #imgList.append(conImg), nameList.append('conImg')
    imgList.append(k_Robert1Img), nameList.append('k_Robert1Img')
    imgList.append(k_Robert2Img), nameList.append('k_Robert2Img')
    imgList.append(k_Sobel1Img), nameList.append('k_Sobel1Img')
    imgList.append(k_Sobel2Img), nameList.append('k_Sobel2Img')
    imgList.append(k_LaplanceImg), nameList.append('k_LaplanceImg')
    imgList.append(k_Prewitt1Img), nameList.append('k_Prewitt1Img')
    imgList.append(k_Prewitt2Img), nameList.append('k_Prewitt2Img')

    plotImagList(imgList,nameList) 
    pass

if __name__=='__main__':
    main()

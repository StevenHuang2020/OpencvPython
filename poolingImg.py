#python3
#Steven image pooling modoule

import cv2.cv2 as cv2 
import matplotlib.pyplot as plt
from ImageBase import *
from mainImagePlot import plotImagList

def poolingImg(img,kernel,maxOrMean=True):
    #print('img.shape', img.shape)
    #print('kernel.shape', kernel.shape)
    kernel_H = kernel.shape[0]
    kernel_W = kernel.shape[1]
    h = int(img.shape[0]/kernel_H)
    w = int(img.shape[1]/kernel_W)
    #print(h,w)
    newImg = np.zeros([h,w], np.uint8)
    for i in range(h):
        for j in range(w):
            roi = img[i*kernel_H:(i+1)*kernel_H, j*kernel_W:(j+1)*kernel_W]
            #print(roi,np.max(roi),np.mean(roi))
            newImg[i,j] = np.max(roi)
            if not maxOrMean:
                newImg[i,j] = np.mean(roi)

    return newImg

def pollingImgDiffKernel(img):
    imgList = []
    nameList = []
    imgList.append(img), nameList.append('Original')

    k=[3,6,8,10,20]
    for i,k in enumerate(k):
        kernel = np.ones((k,k), np.float32)
        poolImg = poolingImg(img,kernel)
        imgList.append(poolImg), nameList.append('poolingImg'+str(i))

    plotImagList(imgList,nameList,True) 

def pollingImgRecurse(img):
    imgList = []
    nameList = []
    imgList.append(img), nameList.append('Original')

    k = 2
    kernel = np.ones((k,k), np.float32)
    maxOrMean=False

    i=0
    poolImg = poolingImg(img,kernel,maxOrMean)
    imgList.append(poolImg), nameList.append('poolingImg'+str(i))
    i+=1

    poolImg = poolingImg(poolImg,kernel,maxOrMean)
    imgList.append(poolImg), nameList.append('poolingImg'+str(i))
    i+=1

    poolImg = poolingImg(poolImg,kernel,maxOrMean)
    imgList.append(poolImg), nameList.append('poolingImg'+str(i))
    i+=1

    poolImg = poolingImg(poolImg,kernel,maxOrMean)
    imgList.append(poolImg), nameList.append('poolingImg'+str(i))
    i+=1

    poolImg = poolingImg(poolImg,kernel)
    imgList.append(poolImg), nameList.append('poolingImg'+str(i))
    i+=1

    plotImagList(imgList,nameList,True) 


def main():
    file = r'./res/Lenna.png' #r'./res/obama.jpg'#
    img = loadImg(file,mode=cv2.IMREAD_GRAYSCALE) # IMREAD_GRAYSCALE IMREAD_COLOR
    img  = OtsuMethodThresHold(img) #binary image
    
    #kernel = np.ones((3,3), np.float32)
    #poolImg = poolingImg(img,kernel)

    #pollingImgDiffKernel(img)
    pollingImgRecurse(img)
    pass

if __name__=='__main__':
    main()

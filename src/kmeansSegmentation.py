#python3 Steven K-Means segmentation
import numpy as np
import cv2
from ImageBase import *
from kmeansModel import KMeansModelTrain,KMeansModel

def drawPointsImg(img, points,color=(255,0,0)):
    newImg = img.copy()
    for center in points:
        cv2.circle(newImg, (center[0],center[1]), radius=1, color=color, thickness=1, lineType=8, shift=0)
    return newImg

def getImageData(img):
    chn = getImagChannel(img)
    Z = img.reshape((-1,chn))
    # convert to np.float32
    Z = np.float32(Z)
    return Z

def KMeansSegmentation(img,k=8):
    Z = getImageData(img)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    print('center=',center)
    res = center[label.flatten()]
    return res.reshape((img.shape)),center

def KMeansSegmentation2(img,k=2):
    chn = getImagChannel(img)
    H,W = getImgHW(img)
    Z = getImageData(img)
    print(Z.shape)
    #k = KMeansModelTrain('image', Z)
    k, centroids, labels = KMeansModel(k, Z)
    
    print('centroids=',centroids)
    labels = labels.reshape((H,W))
    colors = np.random.uniform(0, 255, size=(k, chn))
    print('colors=', colors)
    
    newImg = np.zeros_like(img)
    if chn == 1:
        for i in range(H):
            for j in range(W):
                newImg[i,j] = colors[labels[i,j]]
    else:
        for i in range(H):
            for j in range(W):                
                newImg[i,j,:] = colors[labels[i,j]]
    
    return newImg
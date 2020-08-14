import numpy as np
import cv2

def gaborImg(img,ksize=5,sigma=1, theta=0, lamda=np.pi/4, gamma=0.05):
    kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma)            
    return cv2.filter2D(img, cv2.CV_8UC3, kernel)

def garborFeature(img, ksize=5):#create different feature images
    num=1
    #kernels=[]
    for theta in range(2):
        theta = theta/4.0*np.pi
        for sigma in (1,3):
            for lamda in np.arange(0, np.pi, np.pi/4):
                for gamma in (0.05, 0.5):
                    gabor_label= 'Gabor' + str(num)
                    
                    fimg = gaborImg(img,ksize,sigma, theta, lamda, gamma)                   
                                        
                    print(gabor_label, ': theta=',theta,'sigma=',sigma,'lamda=',lamda,'gamma=',gamma)
                    num += 1
                    yield gabor_label,fimg
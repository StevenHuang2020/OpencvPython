#python3 Steven image smoothing
import cv2

def blurImg(img,ksize=5): #Averaging adjency pixsel 5x5 size kernel
    return cv2.blur(img, (ksize,ksize))

def gaussianBlurImg(img, ksize=5): #Gaussian Blurring
    kernel = cv2.getGaussianKernel(ksize,0)
    #print('gaussian kernel=',kernel)
    return cv2.GaussianBlur(img,(ksize,ksize),0)

def medianBlurImg(img,ksize=5): #Median Blurring
    return cv2.medianBlur(img, ksize)

def bilateralBlurImg(img,d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img,d,sigmaColor,sigmaSpace)

    
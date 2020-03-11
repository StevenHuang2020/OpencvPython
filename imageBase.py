#python3
#Steven Image base operation Class
import cv2.cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

class ImageBase:
    def __init__(self,file,mode=cv2.IMREAD_COLOR):
        cv2.useOptimized()
        self.file = file
        self.image = self.loadImg(file,mode)
  
    def loadImg(self,filename,mode=cv2.IMREAD_COLOR):
        #mode = cv2.IMREAD_COLOR
        #mode = cv2.IMREAD_GRAYSCALE
        #mode = cv2.IMREAD_UNCHANGED
        image = cv2.imread(filename,mode)
        return image
    
    def infoImg(self,str='image:'):
        return(str,'shape:',self.image.shape,'size:',self.image.size,'dtype:','dims=',self.image.ndim,self.image.dtype)
    
    def showimage(self,str='image',autoSize=False):
        flag = cv2.WINDOW_NORMAL
        if autoSize:
            flag = cv2.WINDOW_AUTOSIZE

        cv2.namedWindow(str, flag)
        cv2.imshow(str,self.image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # key = cv2.waitKey(0)
        # #print(key,key & 0xff)
        # if  key & 0xff == 27: #27:ESC   ord('q')
        #     cv2.destroyAllWindows()
        # else:
        #     #print(key)
        #     pass
        return

    def plotImg(self):
        plt.imshow(self.image)
        #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
        plt.show()

    def getImgHW(self):
        return self.image.shape[0],self.image.shape[1]

    def calcAndDrawHist(self, color=[255,255,255]): #color histgram
        hist= cv2.calcHist([self.image], [0], None, [256], [0.0,255.0])
        #print(hist)

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
        histImg = np.zeros([256,256,3], np.uint8) #[256,256,3] [256,256]
        hpt = int(0.9* 256);

        for h in range(256):
            intensity = int(hist[h]*hpt/maxVal)
            #print(h,intensity)
            cv2.line(histImg,(h,256), (h,256-intensity), color)
        return histImg;

    """----------operation------start----------"""
    def binaryImage(self,thresHMin=None,thresHMax=None):
        H, W = getImgHW()
        newImage = self.image.copy()
        for i in range(H):
            for j in range(W):
                if newImage[i,j]<thresHMin:
                    newImage[i,j] = 0
                if newImage[i,j]>thresHMax:
                    newImage[i,j] = 0

        return newImage
    
    def thresHoldImage(self,mode=cv2.THRESH_BINARY):
        newImage = self.image.copy()
        #cv2.THRESH_BINARY
        #cv2.THRESH_BINARY_INV
        #cv2.THRESH_TRUNC
        #cv2.THRESH_TOZERO_INV
        ret,thresh = cv2.threshold(newImage,127,255,mode)
        return thresh

    """----------operation------end------------"""


    """----------operation------start----------"""
    """----------operation------start----------"""
    pass

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

        """#change b g r channel order avoid color not correct when plot"""
        if mode == cv2.IMREAD_COLOR:  
            b,g,r = cv2.split(self.image)       # get b,g,r
            self.image = cv2.merge([r,g,b])     # switch it to rgb
  
    def loadImg(self,filename,mode=cv2.IMREAD_COLOR):
        #mode = cv2.IMREAD_COLOR
        #mode = cv2.IMREAD_GRAYSCALE
        #mode = cv2.IMREAD_UNCHANGED
        image = cv2.imread(filename,mode)
        return image
    
    def infoImg(self,str='image:'):
        return(str,'shape:',self.image.shape,'size:',self.image.size,'dtype:','dims=',self.image.ndim,self.image.dtype)
    
    def getImagChannel(self):
        if self.image.ndim == 3: #color r g b channel
            return 3
        return 1  #only one channel

    def showimage(self,str='image',autoSize=False):
        flag = cv2.WINDOW_NORMAL
        if autoSize:
            flag = cv2.WINDOW_AUTOSIZE

        cv2.namedWindow(str, flag)
        cv2.imshow(str,self.image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
        hpt = int(0.9* 256)

        for h in range(256):
            intensity = int(hist[h]*hpt/maxVal)
            #print(h,intensity)
            cv2.line(histImg,(h,256), (h,256-intensity), color)
        return histImg

    def equalizedHist(self,img=None):
        if img.all() == None:
            return cv2.equalizeHist(self.image.copy())
        else:
            return cv2.equalizeHist(img.copy())

    """----------operation------start----------"""
    def binaryImage(self,thresHMin=0,thresHMax=0):
        """img must be gray"""
        H, W = self.getImgHW()
        newImage = self.image.copy()
        for i in range(H):
            for j in range(W):
                #print(newImage[i,j])
                if newImage[i,j] < thresHMin:
                    newImage[i,j] = 0
                if newImage[i,j] > thresHMax:
                    newImage[i,j] = 255

        return newImage
    
    def thresHoldImage(self,thres=127,mode=cv2.THRESH_BINARY):
        newImage = self.image.copy()
        #cv2.THRESH_BINARY
        #cv2.THRESH_BINARY_INV
        #cv2.THRESH_TRUNC
        #cv2.THRESH_TOZERO_INV
        _, threshold = cv2.threshold(newImage,thres,255,mode)
        return threshold
    
    def OtsuMethodThresHold(self):
        newImage = self.image.copy()
        _, threshold = cv2.threshold(newImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering
        #blur = cv2.GaussianBlur(img,(5,5),0)
        #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return threshold
        
    def thresHoldModel(self,mode=cv2.ADAPTIVE_THRESH_MEAN_C):
        newImage = self.image.copy()
        #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        return cv2.adaptiveThreshold(newImage,255,mode,cv2.THRESH_BINARY,11,2)
    
    
            

    """----------operation------end------------"""


    """----------operation------start----------"""
    """----------operation------start----------"""
    pass

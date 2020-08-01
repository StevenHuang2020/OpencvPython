#python3
#Steven Image base operation Class
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def changeBgr2Rbg(img): #input color img
    if getImagChannel(img) == 3:
        b,g,r = cv2.split(img)       # get b,g,r
        img = cv2.merge([r,g,b])
    return img

def loadImg(file,mode=cv2.IMREAD_COLOR):
    #mode = cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE cv2.IMREAD_UNCHANGED
    try:
        img = cv2.imread(file,mode)
    except:
        print("Load image error,file=",file)
        
    if getImagChannel(img) == 3:
        img = changeBgr2Rbg(img)
    return img

def writeImg(img,filePath):
    cv2.imwrite(filePath,img)

def infoImg(img,str='image:'):
    return print(str,'shape:',img.shape,'size:',img.size,'dims=',img.ndim,'dtype:',img.dtype)

def grayImg(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def getImagChannel(img):
    if img.ndim == 3: #color r g b channel
        return 3
    return 1  #only one channel

def resizeImg(img,NewW,NewH):
    rimg = cv2.resize(img, (NewW,NewH), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC INTER_NEAREST INTER_LINEAR INTER_AREA
    return rimg

def showimage(img,str='image',autoSize=False):
    flag = cv2.WINDOW_NORMAL
    if autoSize:
        flag = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(str, flag)
    cv2.imshow(str,img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def plotImg(img,gray=False):
    if gray:
        plt.imshow(img,cmap="gray")
    else:
        plt.imshow(img)
            
    #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    plt.show()

def getImgHW(img):
    return img.shape[0],img.shape[1]

def distanceImg(img1,img2):
    return np.sqrt(np.sum(np.square(img1 - img2)))

"""-----------------------operation start-------"""
def calcAndDrawHist(img,color=[255,255,255]): #color histgram
    hist= cv2.calcHist([img], [0], None, [256], [0.0,255.0])
    #print(hist)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8) #[256,256,3] [256,256]
    hpt = int(0.9* 256)

    for h in range(256):
        intensity = int(hist[h]*hpt/maxVal)
        #print(h,intensity)
        cv2.line(histImg,(h,256), (h,256-intensity), color)
    return histImg

def equalizedHist(img):
    chn = getImagChannel(img)
    if chn == 1:
        return cv2.equalizeHist(img)
    else:
        newImg = img.copy()
        for n in range(chn):
            newImg[:,:,n] = cv2.equalizeHist(newImg[:,:,n])
        return newImg

def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))

def binaryImage(img,thresH):
    """img must be gray"""
    H, W = getImgHW(img)
    newImage = img.copy()
    for i in range(H):
        for j in range(W):
            #print(newImage[i,j])
            if newImage[i,j] < thresH:
                newImage[i,j] = 0
            else:
                newImage[i,j] = 255

    return newImage

def binaryImage2(img,thresHMin=0,thresHMax=0):
    """img must be gray"""
    H, W = getImgHW(img)
    newImage = img.copy()
    for i in range(H):
        for j in range(W):
            #print(newImage[i,j])
            if newImage[i,j] < thresHMin:
                newImage[i,j] = 0
            if newImage[i,j] > thresHMax:
                newImage[i,j] = 255

    return newImage

def thresHoldImage(img,thres=127,mode=cv2.THRESH_BINARY):
    #mode = cv2.THRESH_BINARY cv2.THRESH_BINARY_INV
    #cv2.THRESH_TRUNC cv2.THRESH_TOZERO_INV
    _, threshold = cv2.threshold(img,thres,255,mode)
    return threshold

def OtsuMethodThresHold(img):
    _, threshold = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold
    
def thresHoldModel(img,mode=cv2.ADAPTIVE_THRESH_MEAN_C):
    #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.adaptiveThreshold(img,255,mode,cv2.THRESH_BINARY,11,2)

def convolutionImg(img,kernel):
    return cv2.filter2D(img,-1,kernel)
    
def colorSpace(img):
    #flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    #print(flags)
    #flag = cv2.COLOR_BGR2GRAY
    flag = cv2.COLOR_BGR2HSV
    return cv2.cvtColor(img, flag)

def flipImg(img,leftRight=True):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    
    newImage = img.copy()
    if leftRight:
        for j in range(W):
            if chn>1:
                for n in range(chn):
                    newImage[:, j, n] = img[:, W-j-1, n]
            else:
                newImage[:, j] = img[:, W-j-1]
    else:
        for i in range(H):
            if chn>1:
                for n in range(chn):
                    newImage[i, :, n] = img[H-i-1, :,n]
            else:
                newImage[i, :] = img[H-i-1, :]

    return newImage

def custGray(img,mean=True):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros((H,W))
    if chn==1:
        return img
    
    if mean:    
        #grayscale = (R + G + B)/3 
        # for i in range(H):
        #     newImage[i, :] = (img[i, :, 0] + img[i, :, 1] + img[i, :, 2])//3
        newImage[:, :] = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])/3
    else:       
        #grayscale = ( (0.3 * R) + (0.59 * G) + (0.11 * B) )
        # for i in range(H):
        #     newImage[i, :] = (img[i, :, 0]*0.3 + img[i, :, 1]*0.59 + img[i, :, 2]*0.11)
        #newImage[:, :] = (img[:, :, 0]*0.3 + img[:, :, 1]*0.59 + img[:, :, 2]*0.11)
        
        #grayscale = ( (0.229 * R) + (0.587 * G) + (0.114 * B) )
        newImage[:, :] = (img[:, :, 0]*0.229 + img[:, :, 1]*0.587 + img[:, :, 2]*0.114)
        
    #newImage = newImage % 255
    return newImage

def differImg(img, flag=0):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros_like(img)
    
    if flag == 0: #leftRight
        for j in range(W-1):
            if chn>1:
                for n in range(chn):
                    newImage[:, j, n] = img[:, j+1, n] - img[:, j, n]
            else:
                newImage[:, j] = img[:, j+1] - img[:, j]
    elif flag == 1: #up down deviation
        for i in range(H-1):
            if chn>1:
                for n in range(chn):
                    newImage[i, :, n] = img[i+1, :, n] - img[i, :, n]
            else:
                newImage[i, :] = img[i+1, :] - img[i, :]
    else:
        for i in range(H-1):
            for j in range(W-1):
                if chn>1:
                    for n in range(chn):
                        a = (img[i, j+1, n] - img[i, j, n])**2
                        b = (img[i+1, j, n] - img[i, j, n])**2                        
                        newImage[i, j, n] = np.sqrt(a+b) #deviations
                else:
                    a = (img[i, j+1] - img[i, j])**2
                    b = (img[i+1, j] - img[i, j])**2                        
                    newImage[i, j] = np.sqrt(a+b) #deviations
                
    return newImage

def reverseImg(img):
    H,W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros_like(img)
    #print(H,W)
    if chn ==1:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = 255-img[i,j]
    else:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = [255-img[i,j,0], 255-img[i,j,1], 255-img[i,j,2]]
    return newImage

def LightImg(img,ratio=1):
    H, W = getImgHW(img)
    chn =getImagChannel(img)
    newImage = img.copy()
    '''
    for i in range(H):
        for j in range(W):
            b = img[i,j,0]*ratio
            g = img[i,j,1]*ratio
            r = img[i,j,2]*ratio
            if b>255:
                b = 255
            if r>255:
                r = 255
            if g>255:
                g = 255

            newImage[i,j]=[b,g,r]
    '''
    '''
    b = img[:,:,0]*ratio
    g = img[:,:,1]*ratio
    r = img[:,:,2]*ratio
    b[b>255] = 255
    r[r>255] = 255
    g[g>255] = 255

    newImage[:,:,0]=b
    newImage[:,:,1]=g
    newImage[:,:,2]=r
    '''
    a = img[:,:,:]*ratio
    a[a>255] = 255
    newImage[:,:,:]=a
    return newImage

def contrastFilterImg(img):#filter fuction to extent [min,max]
    min = np.min(img)
    max = np.max(img)
    
    print('Before contrast:min,max=',min,max)
    
    def f(x):
        #return 100
        #return 1.5*x + 100
        #return f1(x)
        return f3(x)
    
    def f3(x):#Nonlinear Stretching  
        return 10*np.log(x+1) #c*log(x + 1)  c=1
        return np.power(x,2) # c*x^r  c=1,r=0.6
    
    def f2(x): #piecewise (0,0) (100,50) (200,150) (255,255)
        if x<100:
            return 50*x/100
        elif x<200:
            return x-50
        else:
            return 21*x/11 - 200*21/11 + 150
        
    def f1(x): #linear
        v = 255*(x-min)/(max-min)
        if v>255: v = 255
        return v
    
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros_like(img)
    
    if chn == 1:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = f(img[i,j])
    else:  
        for i in range(H):
            for j in range(W):
                b = img[i,j,0]
                g = img[i,j,1]
                r = img[i,j,2]
                newImage[i,j] = [f(b),f(g),f(r)]

    print('After contrast:min,max=',np.min(newImage),np.max(newImage))
    return newImage

def BaoguangImg(img, thres=128):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros_like(img)
    
    def getBGOne(v):
        if v<thres:
            return 255-v
        return v
    
    if chn == 1:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = getBGOne(img[i,j])
    else:
        for i in range(H):
            for j in range(W):
                newImage[i,j] = [getBGOne(img[i,j,0]),getBGOne(img[i,j,1]),getBGOne(img[i,j,2])]
    return newImage

def KuosanImg(img,N=3): #NxN jishu
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = img.copy()
    off = N//2
    for i in range(off,H-off):
        for j in range(off,W-off):
            lst = [img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j-1],img[i,j+1],img[i+1,j-1],img[i+1,j],img[i+1,j+1]]
            #id = random.randint(0,len(lst)-1)
            newImage[i,j] = random.choice(lst) #lst[id]
    return newImage

def meanImg(img):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    #mean = np.sum(img[:,:,:])/(H*W*chn)
    #print('mean = ',mean)
    return np.mean(img)

def deviationImg(img):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    mean = meanImg(img)
    if 1:
        return np.sum((img[:,:,:]-mean)**2)/(H*W*chn)
    else:
        variance = np.sum(img[:,:,:]**2)/(H*W*chn)
        return variance - mean**2
    
def pyramidImg(img): #2x2-->1
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    newImage = np.zeros((H//2, W//2,chn),dtype=np.uint8)
    
    filter=[2,2]
    if chn == 1:
        for i in range(H//2):
            for j in range(W//2):
                startX=j*filter[1]
                stopX = startX + filter[1]
                startY=i*filter[0]
                stopY = startY + filter[0]
                
                newImage[i,j] = np.mean(img[startY:stopY,startX:stopX])
    else:
        for i in range(H//2):
            for j in range(W//2):
                startX=j*filter[1]
                stopX = startX + filter[1]
                startY=i*filter[0]
                stopY = startY + filter[0]
                
                r = np.mean(img[startY:stopY,startX:stopX,0])
                g = np.mean(img[startY:stopY,startX:stopX,1])
                b = np.mean(img[startY:stopY,startX:stopX,2])
                newImage[i,j] = [r,g,b]
       
    return newImage

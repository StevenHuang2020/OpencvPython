#python3
import numpy as np
from ImageBase import *
from mainImagePlot import plotImagList
from videoCap import *
from matplotlib import pyplot as plt
import cv2
from ImageSmoothing import *
from garborFeatures import gaborImg
from matplotHist import plotHistImg

def capColor():
    cap = openCap()
    while(1):
        # Take each frame
        _, frame = cap.read()
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    
def testFlip(img):
    gray = grayImg(img)
    color = colorSpace(img)
    #capColor()
    #thresHold(img)
    #thresHold2(img)
    flip = flipImg(img)
    flip2 = flipImg(img,False)
    grayFlip = flipImg(gray)
    
    ls,nameList = [],[]
    #ls.append(img),nameList.append('Original')
    #ls.append(color),nameList.append('color')
    ls.append(gray),nameList.append('gray')
    ls.append(grayFlip),nameList.append('grayFlip')
    
    ls.append(flip),nameList.append('flip')
    ls.append(flip2),nameList.append('flip2')
   
    plotImagList(ls,nameList,title='Flip image',gray=False)
    
def testGrayImg(img):
    gray = grayImg(img)
    grayMean = custGray(img)
    grayC = custGray(img,False)
    #print(grayC[:1,:5])
    #infoImg(grayMean)
    
    ls,nameList = [],[]
    #ls.append(img),nameList.append('Original')
    ls.append(gray),nameList.append('gray')
    ls.append(grayMean),nameList.append('grayMean')
    ls.append(grayC),nameList.append('grayC')
    
    plotImagList(ls,nameList,title='Customized Gray Image',gray=True)
    
def testDifferImg(img):
    difRL = differImg(img)
    difUD = differImg(img,1)
    gray = grayImg(img)
    difGrayRL = differImg(gray)
    difGrayUD = differImg(gray,1)
    
    ls,nameList = [],[]
    ls.append(difRL),nameList.append('difRL')
    ls.append(difUD),nameList.append('difUD')
    ls.append(difGrayRL),nameList.append('difGrayRL')
    ls.append(difGrayUD),nameList.append('difGrayUD')
    plotImagList(ls, nameList, title='Gradient image',gray=True)
    
def testDiffer2Img(img):
    img = grayImg(img)
    difRL = differImg(img)
    difUD = differImg(img,1)
   
    difRL2 = differImg(difRL)
    difUD2 = differImg(difUD,1)
    
    ls,nameList = [],[]
    ls.append(difRL),nameList.append('difRL')
    ls.append(difUD),nameList.append('difUD')
    ls.append(difRL2),nameList.append('difRL2')
    ls.append(difUD2),nameList.append('difUD2')
    plotImagList(ls, nameList, title='2nd Gradient image',gray=True)

def testDiffer3Img(img):
    #img = grayImg(img)
    difRL = differImg(img)
    difUD = differImg(img,1)
    difAll = differImg(img,2)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('img')
    ls.append(difRL),nameList.append('difRL')
    ls.append(difUD),nameList.append('difUD')
    ls.append(difAll),nameList.append('difAll')
    plotImagList(ls, nameList, title='Gradient tow directions image', gray=True)
    
def testPyramid(img):
    #img = grayImg(img)
    pyramid = pyramidImg(img)
    pyramid2 = pyramidImg(pyramid)
    pyramid4 = pyramidImg(pyramid2)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('img')
    ls.append(pyramid),nameList.append('pyramid')
    ls.append(pyramid2),nameList.append('pyramid2')
    ls.append(pyramid4),nameList.append('pyramid4')
    plotImagList(ls, nameList,title='Pyramid image')
    
def testSmoothing(img):
    blur = blurImg(img,ksize=7)
    gaussian = gaussianBlurImg(img,ksize=7)
    medianBlur = medianBlurImg(img,ksize=7)
    bilateral = bilateralBlurImg(img)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(blur),nameList.append('blur')
    ls.append(gaussian),nameList.append('gaussian')
    ls.append(medianBlur),nameList.append('medianBlur')
    ls.append(bilateral),nameList.append('bilateral')
    
    plotImagList(ls, nameList,title='Smoothing image')
    
def testNoiseImg(img):
    infoImg(img)
    noise = noiseImg(img,50000)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(noise),nameList.append('noise')
    
    plotImagList(ls, nameList,title='Add noise image')
    writeImg(changeRbg2Bgr(noise),r'.\res\LennaNoise.png')
    
def testProjection(img):
    gray = grayImg(img)
    binary = binaryImage(gray,120)
    
    xPixNums, yPixNums = grayProjection(binary)
    print('xPixNums=',xPixNums)
    print('yPixNums=',yPixNums)
    
    # ls,nameList = [],[]
    # ls.append(img),nameList.append('Original')
    # ls.append(gray),nameList.append('gray')
    # ls.append(binary),nameList.append('binary')

    # plotImagList(ls, nameList)
    
    plt.plot(range(len(yPixNums)), yPixNums)
    plt.show()

def testGarbor(img):
    gray = grayImg(img)
    garbo = gaborImg(gray)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(gray),nameList.append('gray')
    ls.append(garbo),nameList.append('garbo')

    plotImagList(ls, nameList,title='Gabor filter')
    
def testEqualizedHistImg(img):
    gray = grayImg(img)
    eqImg = custEqualizedHist(img)
    eqGrayImg = custEqualizedHist(gray)
    #plotHistImg(eqImg)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(eqImg),nameList.append('eqImg')
    ls.append(gray),nameList.append('gray')
    ls.append(eqGrayImg),nameList.append('eqGrayImg')

    plotImagList(ls, nameList,title='Equalized Histogram')
    
def testSubtractImg():
    file1 = r'.\res\Lenna.png' # r'.\res\pngs\out14.png'
    file2 = r'.\res\LennaStar.png' # r'.\res\pngs\out15.png'
    
    img1 = loadImg(file1)
    img2 = loadImg(file2)
    
    img1 = grayImg(img1)
    img2 = grayImg(img2)
    img1 = blurImg(img1)
    img2 = blurImg(img2)
    
    diff = img1-img2
    bDiff = binaryImage(diff,20)
    ls,nameList = [],[]
    ls.append(img1),nameList.append('img1')
    ls.append(img2),nameList.append('img2')
    ls.append(diff),nameList.append('diff')
    ls.append(bDiff),nameList.append('bDiff')
    
    #plotHistImg(diff)
    #plt.hist(diff.ravel(),256,[0,256])
    plotImagList(ls, nameList,gray=True,title='Subtract Image')

def testEdgeImg(img):
    canny = cannyImg(img)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('img')
    ls.append(canny),nameList.append('canny')
    plotImagList(ls, nameList,gray=True,title='Canny edge Image')
    
def testImgSegMask():
    def processMaskImg(img):
        H,W = getImgHW(img)
        chn = getImagChannel(img)
        cl = np.unique(img)
        print(cl)
        colors = np.random.uniform(0, 255, size=(len(cl), chn))
        print('colors=', colors)
        newImg = np.zeros_like(img)
        for i in range(H):
            for j in range(W):
                newImg[i,j,:] = colors[img[i,j,0]]
        return newImg
    
    def maskToOrignimalImg(img,maskImg):
        H,W = getImgHW(img)
        chn = getImagChannel(img)
        #print(img.shape,maskImg.shape)
        cl = np.unique(maskImg)
        colors = np.random.uniform(0, 255, size=(len(cl), chn))
        newImg = img.copy()
        for i in range(H):
            for j in range(W):
                if maskImg[i,j,0] != 0:
                    newImg[i,j,:] = colors[maskImg[i,j,0]]
                
        return newImg
    
    base = r'E:\opencv\project\PedSegmentation\PennFudanMaskAugmentation\res\PennFudanPed\newImages'
    file = r'./res/FudanPed00001.png'
    mask = r'./res/FudanPed00001_mask.png'
    file = r'.\res\FudanPed00059.png'
    mask = r'.\res\FudanPed00059_mask.png'
    #file = r'.\PedSegmentation\res\PennFudanPed\newImages\trainImages\trainPNGImage\FudanPed00019_0_11_363_438.png'
    #mask = r'.\PedSegmentation\res\PennFudanPed\newImages\trainImages\trainPNGImageMask\FudanPed00019_0_11_363_438_mask.png'
    
    # file = base + r'\newMaskFlipping\FudanPed00011_flip.png'
    # mask = base + r'\newMaskFlippingMask\FudanPed00011_flip_mask.png'
    
    # file = base + r'\newMaskScaling\FudanPed00001_scale_1.2894736842105263.png'
    # mask = base + r'\newMaskScalingMask\FudanPed00001_scale_1.2894736842105263_mask.png'
    
    img = loadImg(file)
    imgMask = loadImg(mask)
    
    H,W = getImgHW(img)
    newH,newW = H//2, W//2    
    img = resizeImg(img,newW,newH)
    imgMask = resizeImg(imgMask,newW,newH)
    imgMaskColor = processMaskImg(imgMask)
    maskImg = maskToOrignimalImg(img,imgMask)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('img')
    ls.append(imgMask),nameList.append('imgMask')
    ls.append(imgMaskColor),nameList.append('imgMaskColor')
    ls.append(maskImg),nameList.append('maskImg')
    
    plotImagList(ls, nameList,gray=False,title='Mask Image',showticks=False)
    
def test():
    #https://docs.opencv.org/3.4/d1/d1b/group__core__hal__interface.html
    print(cv2.CV_8U)
    print(cv2.CV_8S)
    print(cv2.CV_8UC2)
    print(cv2.CV_8UC3)
    print(cv2.CV_8SC3)
    
def testBrightnessAndContrast(img):
    ls,nameList = [],[]
    ls.append(img),nameList.append('Orignal')
    
    #test1
    # beta = 50
    # for alpha in np.linspace(0.1,2,5):
    #     ls.append(adjustBrightnessAndContrast(img,alpha,beta)),nameList.append('alpha,beta='+str(alpha)+','+str(beta))
    
    # plotImagList(ls, nameList,gray=False,title='Brightness Image',showticks=False)
    
    #test2
    alpha = 0.6
    for beta in np.linspace(1,200,5):
        ls.append(adjustBrightnessAndContrast(img,alpha,beta)),nameList.append('alpha,beta='+str(alpha)+','+str(beta))
    
    plotImagList(ls, nameList,gray=False,title='Brightness2 Image',showticks=False)
    
def testGammaCorrection(img):
    ls,nameList = [],[]
    ls.append(img),nameList.append('Orignal')
    
    for gamma in np.linspace(0.05,6,11):
        gamma = gamma.round(2)
        ls.append(GammaCorrection(img,gamma)),nameList.append('gamma='+str(gamma))
    
    plotImagList(ls, nameList,gray=False,title='Gamma Correction Image',showticks=False)
    
if __name__ == "__main__":
    img = loadImg(r'./res/Lenna.png') #Lenna.png
    #img = loadImg(r'./res/shudu2.jpg',0)
    #img = loadImg(r'./res/LennaNoise.png')
    #img = loadGrayImg(r'./res/LennaGray.png') #Lenna.png
    #test()
    
    #infoImg(img)
    # infoImg(grayImg(img))
    # infoImg(loadImg(r'./res/LennaGray.png'))
    #testFlip(img)
    #testGrayImg(img)
    #testDifferImg(img)
    #testDiffer2Img(img)
    #testDiffer3Img(img)
    #print('mean,varianceImg=',meanImg(img),varianceImg(img))
    #testPyramid(img)
    #testSmoothing(img)
    #testNoiseImg(img)
    #testProjection(img)
    #testGarbor(img)
    #testEqualizedHistImg(img)
    #testSubtractImg()
    #testEdgeImg(img)
    #testImgSegMask()
    #testBrightnessAndContrast(img)
    testGammaCorrection(img)
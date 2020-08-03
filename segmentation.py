#python3 Steven Otsu binary segmentation
import cv2
from ImageBase import *
from mainImageHist import plotImagAndHist,plotImgHist
from mainImagePlot import plotImagList
import matplotlib.pyplot as plt
from otsuAlgorithm import calculateOtsu,plotHistAndOtsu
from kmeansSegmentation import KMeansSegmentation,drawPointsImg,KMeansSegmentation2

def testOtsu(img):
    gray = grayImg(img)
    hist,fns, thresh = calculateOtsu(gray)
    print('fns=',fns, 'otsuMinThres=', thresh)
    
    fns = np.array(fns)
    fns = fns/100
    plotHistAndOtsu(hist,fns,thresh)
    plotImg(binaryImage(gray, thresh), gray=True)
    
def testKMeans(img):
    #img = grayImg(img)
    km,centers = KMeansSegmentation(img,3)
    #print('centers=',centers)
    kmDraw = drawPointsImg(km,centers,color=(0,0,255))
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(km),nameList.append('km')
    ls.append(kmDraw),nameList.append('kmDraw')
    plotImagList(ls,nameList,title='CV KMeans Segmentation',showticks=False)
    
def testKMeans2(img):
    gray = grayImg(img)
    km = KMeansSegmentation2(img)
    kmGray = KMeansSegmentation2(gray)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(km),nameList.append('km')
    ls.append(gray),nameList.append('gray')
    ls.append(kmGray),nameList.append('kmGray')
    plotImagList(ls,nameList,title='KMeans Segmentation',showticks=False)
    
def testBinaryThres(img):
    gray = grayImg(img)
    _,_, threshOtsu = calculateOtsu(gray)
    thresAuto = autoThresholdValue(gray)
    print('Otsu Thres=',threshOtsu)
    print('Auto Thres=',thresAuto)
    
    bOtsu = binaryImage(gray,threshOtsu)
    bAuto = binaryImage(gray,thresAuto)
    
    ls,nameList = [],[]
    #ls.append(img),nameList.append('Original')
    #ls.append(gray),nameList.append('gray')
    ls.append(bOtsu),nameList.append('bOtsu ' + str(round(threshOtsu,3)))
    ls.append(bAuto),nameList.append('bAuto ' + str(round(thresAuto,3)))
    plotImagList(ls,nameList,title='Threshold get method',gray=True,showticks=False)
    
def main():
    img = loadImg(r'.\res\Lenna.png') #otsus_algorithm.jpg Lenna.png
    #plotImagAndHist(img)
    #print(np.arange(1,10)) #1,2,3,....,9
    #plotImgHist(img)
    #plt.show()
    
    #testOtsu(img)
    #testKMeans(img)
    #testKMeans2(img)
    testBinaryThres(img)
    
    
if __name__ == "__main__":
    main()
        
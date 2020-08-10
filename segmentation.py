#python3 Steven segmentation test
import cv2
import argparse 
from ImageBase import *
from mainImageHist import plotImagAndHist,plotImgHist
from mainImagePlot import *
import matplotlib.pyplot as plt
from otsuAlgorithm import calculateOtsu,plotHistAndOtsu
from kmeansSegmentation import KMeansSegmentation,drawPointsImg,KMeansSegmentation2
from ImageSmoothing import *

def testOtsu(img):
    gray = grayImg(img)
    hist,fns, thresh = calculateOtsu(gray)
    print('fns=',fns, 'otsuMinThres=', thresh)
    
    fns = np.array(fns)
    fns = fns/100
    plotHistAndOtsu(hist,fns,thresh)
    plotImg(binaryImage(gray, thresh), gray=True)
    
def testKMeans(img,k=2): #opencv kmeans
    #img = grayImg(img)
    #km,centers = KMeansSegmentation(img,k)
    #print('centers=',centers)
    #kmDraw = drawPointsImg(km,centers,color=(0,0,255))
    kAll=[3,4,5]
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    for i in kAll:
        km,centers = KMeansSegmentation(img,i)
        ls.append(km),nameList.append('K-Means k='+str(i))
    #ls.append(kmDraw),nameList.append('kmDraw')
    plotImagList(ls,nameList,title='CV K-Means Segmentation',showticks=False)
    
def testKMeansSciKit(img,gray=False): #sci-kit learn kmeans
    if gray:
        img = grayImg(img)
    
    kAll=[2,3,4]
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    
    for i in kAll:
        km = KMeansSegmentation2(img,k=i)
        ls.append(km),nameList.append('K-Means k=' + str(i))
   
    plotImagList(ls,nameList,title='K-Means Segmentation',gray=gray, showticks=False)
    
def testKMeans2(img): #sci-kit learn kmeans
    gray = grayImg(img)
    km = KMeansSegmentation2(img)
    kmGray = KMeansSegmentation2(gray)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('Original')
    ls.append(km),nameList.append('km')
    ls.append(gray),nameList.append('gray')
    ls.append(kmGray),nameList.append('kmGray')
    plotImagList(ls,nameList,title='K-Means Segmentation',showticks=False)
    
def testAutoAlgorithm(img):
    def plotAutoThresh(thres,TList,T1List,T2List):
        ax = plt.subplot(1,1,1)
        ax.set_title('Auto binary Threshold')
        
        ax.plot(range(len(TList)),TList,label='T')
        ax.plot(range(len(T1List)),T1List,label='T1')
        ax.plot(range(len(T2List)),T2List,label='T2')

        ax.hlines(thres,xmin=0, xmax=10,linestyles='dashdot',color='r',label='optimal thresh')
        ax.set_xlabel('Iters')
        ax.set_ylabel('Threshold')
        ax.legend()
        plt.show()
    
    gray = grayImg(img)
    thres,TList,T1List,T2List = autoThresholdValue(gray,False)
    print('thres=',thres)
    print('TList=',len(TList),TList)
    print('T1List=',len(T1List),T1List)
    print('T2List=',len(T2List),T2List)
    plotAutoThresh(thres,TList,T1List,T2List)
    
    plotImg(binaryImage(gray, thres), gray=True)

def testAdaptThreshImg(img):
    gray = grayImg(img)
    adaGray = thresHoldModel(gray)
    plotImg(adaGray, gray=True)
    
def testBinaryThresCompare(img):
    gray = grayImg(img)
    _,_, threshOtsu = calculateOtsu(gray)
    thresAuto,_,_,_ = autoThresholdValue(gray)
    print('Otsu Thres=',threshOtsu)
    print('Auto Thres=',thresAuto)
    
    bOtsu = binaryImage(gray,threshOtsu)
    bAuto = binaryImage(gray,thresAuto)
    adaptThres = thresHoldModel(gray)
    
    ls,nameList = [],[]
    #ls.append(img),nameList.append('Original')
    ls.append(gray),nameList.append('Original')
    ls.append(bOtsu),nameList.append('Otsu\'s Method ' + str(round(threshOtsu,3)))
    ls.append(bAuto),nameList.append('Balanced ' + str(round(thresAuto,3)))
    ls.append(adaptThres),nameList.append('adapative local thres')
    
    plotImagList(ls,nameList,title='Threshold method',gray=True,showticks=False)
    
def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--image', help='image file', required=True)

    return parser.parse_args()

def main():
    args = argCmdParse()
    img = loadImg(args.image)
    #img = loadImg(r'.\res\ab.png') #otsus_algorithm.jpg Lenna.png
    #return plotImg2(img)
    
    img = gaussianBlurImg(img,ksize=3)
    #img = grayImg(img)
    #plotImagAndHist(img,title='Histogram',gray=True,bar=True)
    #plotImagAndHist(binaryImage(img,110),title='Binary Lenna  & Histogram',gray=True)
    
    #print(np.arange(1,10)) #1,2,3,....,9
    #plotImgHist(img),plt.show()
    
    #testOtsu(img)
    testKMeans(img)
    #testKMeans2(img)
    #testKMeansSciKit(img,gray=False)
    #testAutoAlgorithm(img)
    #testAdaptThreshImg(img)
    #testBinaryThresCompare(img)
    
    
if __name__ == "__main__":
    main()
        
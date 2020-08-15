#python3
#Steven image histgram display modoule
import cv2 #pip install opencv-python
import matplotlib.pyplot as plt

from ImageBase import *
from matplotHist import plotHistImg, getImgHist, getImgHist256Img

def plotImagHistListImgHist256(imgList,title):
    nImg = len(imgList)
    nColumn = 3
    plt.figure().suptitle(title, fontsize="x-large")
    for n in range(nImg):
        img = imgList[n]
        plt.subplot(nImg, nColumn, n*nColumn + 1) #raw img    
        plt.imshow(img)

        plt.subplot(nImg, nColumn, n*nColumn + 2) #hist of raw img   
        plotImgHist(img)

        plt.subplot(nImg, nColumn, n*nColumn + 3) #hist256 of raw img   
        plotImgHist256Img(img)

    #plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotImagHistListImg(imgList,title=''):
    nImg = len(imgList)
    nColumn = 2
    plt.figure().suptitle(title, fontsize="x-large")
    for n in range(nImg):
        img = imgList[n]
        plt.subplot(nImg, nColumn, n*nColumn + 1) #raw img    
        plt.imshow(img)

        plt.subplot(nImg, nColumn, n*nColumn + 2) #hist of raw img   
        plotImgHist(img)

    #plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotImgHist(img,bar=False):
    color = ('b','g','r')
    hists = getImgHist(img)
    cn = 0
    if bar: #bar chart
        for i in hists:
            #print('len=',type(i),i.shape,len(i))
            plt.bar(np.arange(len(i)),i.ravel(),color = color[cn])
            cn+=1
    else: #plot
        for i in hists:
            plt.plot(i,color = color[cn])
            #plt.xlim([0,256])
            cn+=1

def plotImgHist256Img(img):
    #color = ('b','g','r')
    hist256Img = getImgHist256Img(img)
    for i in hist256Img:
        plt.imshow(i)
        break

def plotImagAndHist4(img):  #must color img
    imgGray = grayImg(img)

    imgList=[]
    imgList.append(img)
    imgList.append(imgGray)
    return plotImagHistListImg(imgList,'Image&Gray Histogram')

def plotImagAndHist(img,gray=False,title='',grid=False,bar=False):
    plt.figure().suptitle(title, fontsize="x-large")
    plt.subplot(1, 2, 1)   
    if gray: 
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
        
    plt.subplot(1, 2, 2)
    plotImgHist(img,bar=bar)

    plt.grid(grid)
    plt.tight_layout()
    plt.show()

def plotEqHist4(img):
    imgList=[]
    imgGray = grayImg(img)
    imgList.append(img)
    imgList.append(equalizedHist(img))
    imgList.append(imgGray)
    imgList.append(equalizedHist(imgGray))

    #plotImagHistListImg(imgList,'Image&Gray Histogram')
    plotImagHistListImgHist256(imgList,'Image&Gray Histogram All')
    
def plotHinstAndDerivative(img):
    hists = getImgHist(grayImg(img))
    hists = hists[0].ravel()
    length = len(hists)
    #print('Hist=',length,hists)
    
    derivative = np.zeros((length,))
    for i in range(length-1):
        derivative[i] = hists[i+1]-hists[i]
    #print('derivative =',derivative)
    
    plt.plot(range(length),hists,label='hinst')
    plt.plot(range(length),derivative,label='derivative')
    plt.legend()
    plt.show()
    
def main():
    file = './res/Lenna.png' #r'./res/obama.jpg'# Lenna.png PIA13843.tif
    img = loadImg(file) # IMREAD_GRAYSCALE IMREAD_COLOR
    infoImg(img)
    #showimage(img,autoSize=False)
    #plotImg(img)

    #showimage(calcAndDrawHist(img))
    #showimage(thresHoldImage(img))
       
    #plotHistImg(img)
    #plotImagAndHist(img)
    #plotImagAndHist4(img)
    #plotEqHist4(img)
    #plotHinstAndDerivative(img)
    
    lennaCEq = equalizedHist(img)
    lennaGEq = equalizedHist(grayImg(img))
    writeImg(lennaCEq,r'.\res\LennaCEq.png')
    writeImg(lennaGEq,r'.\res\LennaGEq.png')
    
if __name__=='__main__':
    main()

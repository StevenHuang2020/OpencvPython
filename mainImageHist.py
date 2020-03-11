#python3
#Steven image histgram display modoule
import cv2.cv2 as cv2 #pip install opencv-python
import matplotlib.pyplot as plt

from imageBase import ImageBase
from matplotHist import plotHistImg,getHistImg

def plotImgHist(img):
    color = ('b','g','r')
    hists = getHistImg(img)
    cn = 0
    for i in hists:
        plt.plot(i,color = color[cn])
        plt.xlim([0,256])
        cn+=1

def plotImagAndHist4(img):
    b,g,r = cv2.split(img)       # get b,g,r
    img = cv2.merge([r,g,b])     # switch it to rgb

    plt.subplot(2, 2, 1)    
    plt.imshow(img)

    plt.subplot(2, 2, 2)
    plotImgHist(img)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 2, 3)    
    plt.imshow(imgGray)

    plt.subplot(2, 2, 4)
    plotImgHist(imgGray)

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    pass

def plotImagAndHist(img):
    b,g,r = cv2.split(img)       # get b,g,r
    img = cv2.merge([r,g,b])     # switch it to rgb

    plt.subplot(1, 2, 1)    
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plotImgHist(img)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def showimage(img,str='image',autoSize=False):
    flag = cv2.WINDOW_NORMAL
    if autoSize:
        flag = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(str, flag) #cv2.WINDOW_NORMAL)
    cv2.imshow(str,img)
   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    file = r'./res/Lenna.png'
    img = ImageBase(file,mode=cv2.IMREAD_COLOR) # IMREAD_GRAYSCALE
    print(img.infoImg())
    #img.showimage(autoSize=False)
    #img.plotImg()
    #showimage(img.calcAndDrawHist())
    #showimage(img.thresHoldImage())
    #img.showimage()
    
    #plotHistImg(img.image)
    #plotImagAndHist(img.image)
    plotImagAndHist4(img.image)

if __name__=='__main__':
    main()

#python3
#Steven image threshold modoule
import cv2.cv2 as cv2 #pip install opencv-python
import matplotlib.pyplot as plt
from imageBase import ImageBase
from mainImagePlot import plotImagList

def showimage(img,str='image',autoSize=False):
    flag = cv2.WINDOW_NORMAL
    if autoSize:
        flag = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(str, flag) #cv2.WINDOW_NORMAL)
    cv2.imshow(str,img)
   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    file = r'./res/obama.jpg'#'./res/Lenna.png' #
    img = ImageBase(file,mode=cv2.IMREAD_GRAYSCALE) # IMREAD_GRAYSCALE IMREAD_COLOR
    #print(img.infoImg())
    img1 = img.binaryImage(thresHMin=50,thresHMax=150)
    img2 = img.binaryImage(thresHMin=50,thresHMax=100)
    img3 = img.thresHoldImage(mode = cv2.THRESH_BINARY)
    img4 = img.OtsuMethodThresHold()
    img5 = img.thresHoldModel(mode = cv2.ADAPTIVE_THRESH_MEAN_C)
    img6 = img.thresHoldModel(mode = cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    imgList = []
    nameList = []
    imgList.append(img.image), nameList.append('Original')
    imgList.append(img1), nameList.append('thrHMin')
    imgList.append(img2), nameList.append('thrHMin')
    imgList.append(img3), nameList.append('thrImg_Binary')
    imgList.append(img4), nameList.append('OtsuMethod')
    imgList.append(img5), nameList.append('thr_Mean')
    imgList.append(img6), nameList.append('thr_Gaussian')
    plotImagList(imgList,nameList) 

    
if __name__=='__main__':
    main()

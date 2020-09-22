#python3 Steven image dither
#https://en.wikipedia.org/wiki/Dither
import cv2
import numpy as np
from ImageBase import *
from mainImagePlot import plotImagList

def thresHold(v):
        if v > 255:
            v = 255
        elif v < 0:
            v = 0
        return v
    
def AverageDithering(img):
    return binaryImage(img,100)

def Floyd_SteinbergDithering(img,random=False):
    H,W = getImgHW(img)
    #chn = getImagChannel(img)
    newImg = img.copy()
    for i in range(1,H-1):
        for j in range(1,W-1):
            oldpixel = newImg[i,j]
            if random:
                # rX = np.random.randint(W)
                # rY = np.random.randint(H)
                # oldpixel = newImg[rY,rX]
                oldpixel = np.random.randint(256)
                
            new_value = nearst_palette_color(oldpixel)
            # new_value = 0
            # if (oldpixel > 128) :
            #     new_value = 255
            
            quant_error = oldpixel - new_value
            newImg[i+1, j] = thresHold(newImg[i+1, j] + quant_error * 7 / 16)
            newImg[i-1, j+1] = thresHold(newImg[i-1, j+1] + quant_error * 3 / 16)
            newImg[i, j+1] = thresHold(newImg[i, j+1] + quant_error * 5 / 16)
            newImg[i+1, j+1] = thresHold(newImg[i+1, j+1] + quant_error * 1 / 16)
    return newImg
  
def nearst_palette_color(c,N=2):
    palette = np.arange(0, 256, 255//(N-1))
    i = c*N //255
    #print('i,palette=',i,palette,palette[i])
    return palette[i]

def OrderDithering(img,N=2,random=False):
    H,W = getImgHW(img)
    newImg = img.copy()
    for i in range(1,H-1):
        for j in range(1,W-1):
            oldpixel = newImg[i,j]
            if random:
                oldpixel = np.random.randint(256)
                
            newImg[i, j] = nearst_palette_color(oldpixel,N)
           
    return newImg 

def testFloydDitheringImg(img):
    img = grayImg(img)
    #img = binaryImage(img,100)
    fImg = Floyd_SteinbergDithering(img)
    #averImg = AverageDithering(img)
    #frImg = Floyd_SteinbergDithering(img,True)
    
    ls,nameList = [],[]
    ls.append(img),nameList.append('img')
    ls.append(fImg),nameList.append('fImg')
    #ls.append(frImg),nameList.append('frImg')
    #ls.append(averImg),nameList.append('averImg')
       
    plotImagList(ls, nameList,gray=True,title='Dithering Image',showticks=False)
   
def testOrderDitheringImg(img):
    img = grayImg(img)
    oImg = OrderDithering(img)
    o4Img = OrderDithering(img,N=4)
    o8Img = OrderDithering(img,N=8)
   
    ls,nameList = [],[]
    ls.append(img),nameList.append('img')
    ls.append(oImg),nameList.append('oImg')
    ls.append(o4Img),nameList.append('o4Img')
    ls.append(o8Img),nameList.append('o8Img')

    plotImagList(ls, nameList,gray=True,title='Order Dithering Image',showticks=False)
     
def main():
    img = loadImg(r'./res/Lenna.png') #Lenna.png
    #testFloydDitheringImg(img)
    #nearst_palette_color(120,2)
    testOrderDitheringImg(img)
    
if __name__=='__main__':
    main()
    
#python3
#Steven image mosaic modoule

import cv2.cv2 as cv2 
import matplotlib.pyplot as plt
from ImageBase import *
from mainImagePlot import plotImagList

def mosaicImg(x,y,h,w,img): #all block random
    #h = 30
    #w = 132
    #roi = img[252:252+h, 242:242+w]
    roi = img[y:y+h, x:x+w]
    resImg = roi.copy()
    #block = np.zeros((5,6), np.float32)
    #block = np.zeros((3,3), np.float32)
    block = np.zeros((15,12), np.float32)
    #block = np.zeros((10,12), np.float32)

    block_h = block.shape[0]
    block_w = block.shape[1]

    ag = np.arange(int(h*w/(block_h*block_w)))
    np.random.shuffle(ag)
    print('len=',len(ag))
    for i,blk in enumerate(ag):
        H = int(np.floor(i / (w/block_w)))
        W = int(i % (w/block_w))

        rH = int(np.floor(blk / (w/block_w)))
        rW = int(blk % (w/block_w))
        print('dst=(',H,W,')','src=(',rH,rW,')')

        resImg[H*block_h:H*block_h+block_h, W*block_w:W*block_w+block_w] = roi[rH*block_h:rH*block_h+block_h, rW*block_w:rW*block_w+block_w]
    return resImg

def mosaicImg2(x,y,h,w,img): #random in block
    roi = img[y:y+h, x:x+w] #
    resImg = roi.copy()
    #block = np.zeros((5,6)) #block size must according to h and w
    #block = np.zeros((3,3))
    #block = np.zeros((6,12))
    #block = np.zeros((9,12))
    block = np.zeros((15,12))
    #block = np.zeros((h,w))

    block_h = block.shape[0]
    block_w = block.shape[1]

    blockNum = int(h*w/(block_h*block_w))
    for i in range(blockNum):
        H = int(np.floor(i / (w/block_w)))
        W = int(i % (w/block_w))

        res = randomBolckImg(roi[H*block_h:H*block_h+block_h, W*block_w:W*block_w+block_w])
        resImg[H*block_h:H*block_h+block_h, W*block_w:W*block_w+block_w] = res
    return resImg

def randomBolckImg(img):
    h,w = img.shape
    resImg = img.copy()
    
    ag = np.arange(h*w)
    np.random.shuffle(ag)
    for i,blk in enumerate(ag):
        H = int(np.floor(i / w))
        W = int(i % w)
        rH = int(np.floor(blk / w))
        rW = int(blk % w)
        resImg[H,W] = resImg[rH,rW]
    return resImg

def mosaicImg3(x,y,h,w,img): #all block random
    roi = img[y:y+h, x:x+w]
    resImg = roi.copy()
    #block = np.zeros((5,6), np.float32)
    #block = np.zeros((3,3), np.float32)
    #block = np.zeros((9,12), np.float32)
    block = np.zeros((15,12), np.float32)

    block_h = block.shape[0]
    block_w = block.shape[1]

    len = int(h*w/(block_h*block_w))
    for i in range(len):
        H = int(np.floor(i / (w/block_w)))
        W = int(i % (w/block_w))

        value = np.mean(roi[H*block_h:H*block_h+block_h, W*block_w:W*block_w+block_w])
        #value = np.max(roi[H*block_h:H*block_h+block_h, W*block_w:W*block_w+block_w])
        #value = np.min(roi[H*block_h:H*block_h+block_h, W*block_w:W*block_w+block_w])

        resImg[H*block_h:H*block_h+block_h, W*block_w:W*block_w+block_w] = value
    return resImg

def main():
    file = r'./res/Lenna.png' #r'./res/obama.jpg'#
    img = loadImg(file,mode=cv2.IMREAD_GRAYSCALE) # IMREAD_GRAYSCALE IMREAD_COLOR
        
    imgList = []
    nameList = []

    x,y = 242,245
    h,w = 45,120
    #mosaic = mosaicImg(x,y,h,w,img)
    #mosaic = mosaicImg2(x,y,h,w,img)
    mosaic = mosaicImg3(x,y,h,w,img)

    imgList.append(img), nameList.append('Original')

    mImg = img.copy()
    mImg[y:y+h, x:x+w] = mosaic
    imgList.append(mImg), nameList.append('mosaicImg')

    plotImagList(imgList,nameList,True) 
    pass

if __name__=='__main__':
    main()

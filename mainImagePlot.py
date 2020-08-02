#python3
#Steven 11/03/2020 image plot modoule

import matplotlib.pyplot as plt
from common import getRowAndColumn
from ImageBase import *

def plotImagList(imgList, nameList, title='', gray=False, showticks=True):
    nImg = len(imgList)
    nRow,nColumn = getRowAndColumn(nImg)
    
    plt.figure().suptitle(title, fontsize="x-large")
    for n in range(nImg):
        img = imgList[n]
        ax = plt.subplot(nRow, nColumn, n + 1)
        ax.title.set_text(nameList[n])
        if gray:
            plt.imshow(img,cmap="gray")
        else:
            plt.imshow(img)
        
        if not showticks:
            ax.set_yticks([])
            ax.set_xticks([])
    #plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    file = r'./res/obama.jpg'#'./res/Lenna.png' #
    img = loadImg(file,mode=cv2.IMREAD_GRAYSCALE) # IMREAD_GRAYSCALE IMREAD_COLOR
    infoImg(img)
    img = binaryImage2(img,thresHMin=50,thresHMax=150)
    showimage(img)
    pass

if __name__=='__main__':
    main()

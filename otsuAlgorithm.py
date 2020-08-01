#python3 Steven Otsu binary segmentation
import cv2
from ImageBase import *
from mainImageHist import plotImagAndHist,plotImgHist
import matplotlib.pyplot as plt

class OtsuAlgorithm:
    def __init__(self, img):
        self.img = img
        self.hist = cv2.calcHist([img],[0],None,[256],[0,256])
        self.fn_min = np.inf
        self.thresh = -1
        self.fns = []
        self.calculateOtsu()
        print('len(hist)=',len(self.hist),len(self.fns))
        
    def calculateOtsu(self):
        hist_norm = self.hist.ravel()/self.hist.max()
        Q = hist_norm.cumsum()
        bins = np.arange(256)

        for i in range(1,256):
            p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
            q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
            b1,b2 = np.hsplit(bins,[i]) # weights

            # finding means and variances
            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

            # calculates the minimization function
            fn = v1*q1 + v2*q2
            self.fns.append(fn)
            if fn < self.fn_min:
                self.fn_min = fn
                self.thresh = i

    
def plotHistAndOtsu(hist,otsu,thre):
    ax = plt.subplot(1,1,1)
    ax.set_title('Otsu')
    ax.plot(range(len(hist)),hist,label='hist')
    ax.plot(range(len(otsu)),otsu,label='otsu')
    ax.vlines(thre,ymin=0, ymax=max(np.max(hist),np.max(otsu)),linestyles='dashdot',color='r',label='optimal thresh')
    ax.set_xlabel('pixsel')
    ax.set_ylabel('Hist&Otsu')
    ax.legend()
    plt.show()
    
def main():
    img = loadImg(r'.\res\cap58.jpg') #otsus_algorithm.jpg Lenna.png
    #plotImagAndHist(img)
    print(np.arange(1,10))
    #plotImgHist(img)
    #plt.show()
    gray = grayImg(img)
    otsu = OtsuAlgorithm(gray)
    print('fns=',otsu.fns,'otsuMin=',otsu.fn_min, 'otsuMinThres=', otsu.thresh)
    
    fns = np.array(otsu.fns)
    fns = fns/100
    plotHistAndOtsu(otsu.hist,fns,otsu.thresh)
    
    plotImg(binaryImage(gray, otsu.thresh), gray=True)
    
if __name__ == "__main__":
    main()
        
# python3
# https://en.wikipedia.org/wiki/Discrete_Fourier_transform
import cv2
import numpy as np
# from matplotlib import pyplot as plt
from ImageBase import loadImg
from mainImagePlot import plotImagList


def dftFft(a):
    f = np.fft.fft2(a)
    return np.fft.fftshift(f)


def dftFftBack(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    return np.fft.ifft2(f_ishift)


def dftImg():
    file = r'./res/Lenna.png'  # './res/Lenna.png' #
    # IMREAD_GRAYSCALE IMREAD_COLOR
    img = loadImg(file, mode=cv2.IMREAD_GRAYSCALE)

    fshift = dftFft(img)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    '''handle shift, then back img'''
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    imgList = []
    nameList = []
    imgList.append(img), nameList.append('Original')
    imgList.append(magnitude_spectrum), nameList.append('magnitude_spectrum')

    sizes = [3, 5, 10, 20]
    for size in sizes:
        fshift[crow - size:crow + size, ccol - size:ccol + size] = 0
        img_back = np.real(dftFftBack(fshift))
        imgList.append(img_back), nameList.append('img_back_' + str(size))

    plotImagList(imgList, nameList, True)


def dftTest():
    a = np.array([[1, 2, 3]])
    fshift = dftFft(a)
    print(type(fshift))
    print('fshift = ', fshift)
    print('abs(fshift) = ', abs(fshift))
    print('log = ', np.log(np.abs(fshift)))

    back = dftFftBack(fshift)
    print('back=', back)
    print(abs(back))
    pass


def main():
    # dftTest()
    dftImg()


if __name__ == '__main__':
    main()

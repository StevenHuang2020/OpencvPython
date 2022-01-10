# python3
import cv2
import numpy as np
import operator
from commonPath import *
from ImageBase import *
from ImageSmoothing import *


class ImageU:
    def __init__(self, file):
        self.file = file
        self.img = loadImg(file)
        self.mean = np.mean(self.img)

    def __str__(self):
        return self.file + ' ' + str(self.mean)


class ImageUList:
    def __init__(self):
        self.ImageList = []

    def sort(self):
        self.ImageList = sorted(self.ImageList, key=operator.attrgetter('mean'))

    def add(self, file):
        try:
            image = ImageU(file)
            if not self.hasSameMean(image.mean):
                self.ImageList.append(image)
        except BaseException:
            print('load image error,', file)

    def hasSameMean(self, mean):
        for i in self.ImageList:
            if i.mean == mean:
                return True
            if abs(i.mean - mean) < 0.1:
                return True
        return False

    def print(self):
        print('total:', len(self.ImageList))
        for i in self.ImageList:
            print(i)

    def getImage(self, mean):
        if len(self.ImageList) <= 0:
            return None

        min = self.ImageList[0].mean
        max = self.ImageList[-1].mean
        N = len(self.ImageList)
        if mean < min:
            mean = min
        if mean > max:
            mean = max
        x = (mean - min) * (N - 1) / (max - min)

        return self.ImageList[int(x)].img


def mosaicImg(img, blockSize, sources):  # all block random
    h, w = getImgHW(img)
    resImg = img.copy()

    block_h = blockSize[0]
    block_w = blockSize[1]

    blockNum = int(h * w / (block_h * block_w))
    for i in range(blockNum):
        H = int(np.floor(i / (w / block_w)))
        W = int(i % (w / block_w))

        mean = np.mean(img[H * block_h:H * block_h + block_h, W * block_w:W * block_w + block_w])
        source = sources.getImage(mean)
        source = resizeImg(source, block_w, block_h)
        # print(source.shape)
        try:
            resImg[H * block_h:H * block_h + block_h, W * block_w:W * block_w + block_w] = source
        except BaseException:
            print(H, W, H * block_h, H * block_h + block_h, W * block_w, W * block_w + block_w)
    return resImg


def getSourceImages():
    path = r'.\res'  # r'.\PennFudanAugmentation\res\PennFudanPed\PNGImages'
    images = ImageUList()
    for i in pathsFiles(path, 'jpg bmp png', True):
        images.add(i)
    images.sort()
    images.print()
    return images


def main():
    sources = getSourceImages()

    file = r'.\res\my.png'  # r'.\res\Lenna.png' #
    img = loadImg(file)
    img = blurImg(img, 3)

    res = mosaicImg(img, (2, 2), sources)
    print('res.shape=', res.shape)
    showimage(res)
    writeImg(res, r'.\res\Lena_mos.png')


if __name__ == '__main__':
    main()

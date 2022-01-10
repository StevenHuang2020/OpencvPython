import sys
sys.path.append('..')

import pandas as pd
from ImageBase import cannyImg, grayImg, loadGrayImg
from skimage.filters import roberts, sobel, scharr, prewitt, gaussian, laplace, farid, median
from FeatureExtract.imageFeatures import garborFeature
from mainImagePlot import plotImagList


def makeImageFeatures(img):
    df = pd.DataFrame()

    img1 = img.reshape(-1)
    print(img.shape, img1.shape)
    df['Original Image'] = img1

    # first set -Gabor features
    for (name, feature) in garborFeature(img):
        df[name] = feature.reshape(-1)

    # Canny edge
    df['Canny Edge'] = cannyImg(img).reshape(-1)

    df['Roberts'] = roberts(img).reshape(-1)
    df['Sobel'] = sobel(img).reshape(-1)
    df['Scharr'] = scharr(img).reshape(-1)
    df['Prewitt'] = prewitt(img).reshape(-1)

    df['Gaussian s3'] = gaussian(img, sigma=3).reshape(-1)
    df['Gaussian s5'] = gaussian(img, sigma=5).reshape(-1)
    df['Gaussian s7'] = gaussian(img, sigma=7).reshape(-1)
    df['Lplace'] = laplace(img).reshape(-1)
    df['Farid'] = farid(img).reshape(-1)
    df['Median'] = median(img).reshape(-1)
    return df


def makeDb(img, mask, dbFile):
    df = makeImageFeatures(img)

    label = mask.reshape(-1)
    df['Label'] = label
    print(df.head())
    df.to_csv(dbFile, index=False)


def loadDb(file):
    df = pd.read_csv(file)

    Y = df['Label'].values
    X = df.drop(columns=['Label'])
    # print(X.head())
    return X, Y


def showImageFeatures(img):
    img = grayImg(img)
    ls, nameList = [], []
    # ls.append(cannyImg(img)),nameList.append('Canny')
    # ls.append(roberts(img)),nameList.append('roberts')
    # ls.append(sobel(img)),nameList.append('sobel')
    # ls.append(scharr(img)),nameList.append('scharr')

    # ls.append(prewitt(img)),nameList.append('prewitt')
    # ls.append(gaussian(img)),nameList.append('gaussian')
    # ls.append(laplace(img)),nameList.append('laplace')
    # ls.append(farid(img)),nameList.append('farid')

    for (name, feature) in garborFeature(img):
        ls.append(feature), nameList.append(name)

    plotImagList(ls, nameList, gray=True, title='', showticks=False)


def main():
    file = r'.\res\FudanPed00001.png'
    maskFile = r'.\res\FudanPed00001_mask.png'
    dbFile = r'.\res\FudanPed00001.csv'

    # file = r'..\res\Lenna.png'
    # img = loadImg(file)
    img = loadGrayImg(file)
    mask = loadGrayImg(maskFile)
    makeDb(img, mask, dbFile)
    # showImageFeatures(img)
    # loadDb(dbFile)


if __name__ == '__main__':
    main()

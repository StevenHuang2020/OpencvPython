# python3 Steven Generate interesting images
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ImageBase import grayImg, loadImg, getImagChannel, getImgHW
from ImageBase import infoImg, textImg, reverseImg, binaryImage, resizeImg, cannyImg
from mainImagePlot import plotImagList


def DrawCustText(img, str, loc=(0, 0)):
    fontpath = 'consola.ttf'
    b, g, r, a = 0, 255, 0, 0
    # font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    font = ImageFont.truetype(fontpath, 30)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # draw.text((loc), "Hello!", font=font, fill=(b, g, r, a))
    draw.text((loc), str)
    img = np.array(img_pil)
    return img


def genImg(H, W, chn=1, white=True):
    if chn == 3:
        img = np.zeros([H, W, chn], dtype=np.uint8)
    else:
        img = np.zeros([H, W], dtype=np.uint8)

    if white:
        img += 255
    return img


def addBackgroundGrayImg(img, step=6):
    H, W = getImgHW(img)
    chn = getImagChannel(img)
    for j in range(0, W, step):
        img[:, j] = 0  # black


def getTextImgStr(str, H=60, W=100, font=cv2.FONT_HERSHEY_SIMPLEX):
    img = genImg(H, W, 1)

    # print(img)
    imgText = textImg(img, str, fontFace=font, fontScale=1)
    infoImg(imgText)

    textArt = getBinaryTextImgStr(reverseImg(imgText), char=' ', elseChar='#')
    # print(textArt)
    return textArt


def generateArtText(str='Hello'):  # get text from binary text image
    textArt = getTextImgStr(str)
    imgArtText = genImg(350, 450)
    if 0:
        y0, dy = 5, 15
        font = cv2.FONT_HERSHEY_PLAIN  # cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(textArt.split('\n')):
            y = y0 + i * dy
            imgArtText = textImg(imgArtText, line, loc=(50, y), fontFace=font, fontScale=1)
    else:
        imgArtText = DrawCustText(imgArtText, textArt, loc=(5, 5))

    ls, nameList = [], []
    # ls.append(img),nameList.append('img')
    # ls.append(imgText),nameList.append('imgText')
    # ls.append(bImgText),nameList.append('bImgText')
    ls.append(imgArtText), nameList.append('imgArtText')
    plotImagList(ls, nameList, gray=True, title='Art Text Image')


def getBinaryTextImgStr(img, thresh=100, char='#', elseChar=' '):
    img = grayImg(img)
    img = binaryImage(img, thresh)
    H, W = getImgHW(img)

    def finTextLoc(img):
        hStart, hStop = 0, 0
        wStart, wStop = 0, 0

        for i in range(H):
            if np.sum(img[i, :]) > 0:
                hStart = i - 1
                break
        for i in range(H):
            if np.sum(img[H - i - 1, :]) > 0:
                hStop = H - i + 1
                break
        for j in range(W):
            if np.sum(img[:, j]) > 0:
                wStart = j - 1
                break
        for j in range(W):
            if np.sum(img[:, W - j - 1]) > 0:
                wStop = W - j + 1
                break
        if hStart < 0:
            hStart = 0
        if hStop > H:
            hStop = H
        if wStart < 0:
            wStart = 0
        if wStop > W:
            wStop = W

        return hStart, hStop, wStart, wStop

    lines = []
    if 1:
        hStart, hStop, wStart, wStop = finTextLoc(img)
        print('hStart,hStop,wStart,wStop=', hStart, hStop, wStart, wStop)
        for i in range(hStart, hStop):
            for j in range(wStart, wStop):
                if img[i, j]:
                    lines.append(elseChar)
                else:
                    lines.append(char)
            lines.append('\n')
        return (''.join(lines))
    else:
        for i in range(H):
            for j in range(W):
                if img[i, j]:
                    lines.append(elseChar)
                else:
                    lines.append(char)
            lines.append('\n')
        return (''.join(lines))


def getGrayImageToStr(img):
    img = grayImg(img)
    H, W = getImgHW(img)
    print(H, W)
    ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

    def get_char(gray, alpha=256):
        if alpha == 0:
            return ' '
        length = len(ascii_char)
        # gray=int(0.2126*r+0.7152*g+0.0722*b)
        # unit=(256.0+1)/length
        # print('len,index,gray=',length,int(gray*length/256),gray)
        return ascii_char[int(gray * length / 256)]

    txt = ""
    for i in range(H):
        for j in range(W):
            txt += get_char(img[i, j])
        txt += '\n'

    print(len(txt), len(txt) / H)
    return txt


def getImageCharatStr(img):
    strPhoto = getGrayImageToStr(img)
    # print(strPhoto)

    if 0:  # only photo
        imgArtText = genImg(1070, 840)
        y0, dy = 2, 6
        for i, line in enumerate(strPhoto.split('\n')):
            y = y0 + i * dy
            imgArtText = DrawCustText(imgArtText, line, loc=(2, y))
    elif 1:  # only text
        # showText = 'Welcome!'
        # showText = 'Hi, there! \n Welcome \n to my \n channel. \n My name \n is Steven.'
        showText = 'Welcome,I\'m Steven.'
        # strText = getTextImgStr('Welcome! \n My name is StevenHuang',H=80,W=200)

        imgArtText = genImg(250, 840 + 1060)
        addBackgroundGrayImg(imgArtText)

        offsetX = 0
        offsetY = 0
        for i, lineStr in enumerate(showText.split('\n')):
            y0 = offsetY
            dy = 10
            str = getTextImgStr(lineStr, H=80, W=800)
            for j, line in enumerate(str.split('\n')):
                y = y0 + j * dy
                imgArtText = DrawCustText(imgArtText, line, loc=(offsetX, y))
            offsetY += j * dy

    elif 0:  # photo plus text
        # showText = 'Welcome!'
        showText = 'Hi, there! \n Welcome \n to my \n channel. \n My name \n is Steven.'
        # strText = getTextImgStr('Welcome! \n My name is StevenHuang',H=80,W=200)

        imgArtText = genImg(1070, 840 + 1000)
        addBackgroundGrayImg(imgArtText)

        y0, dy = 2, 6
        for i, line in enumerate(strPhoto.split('\n')):
            y = y0 + i * dy
            imgArtText = DrawCustText(imgArtText, line, loc=(2, y))

        offsetX = 840 + 8
        offsetY = 20
        for i, lineStr in enumerate(showText.split('\n')):
            y0 = 2 + offsetY
            dy = 6
            str = getTextImgStr(lineStr, H=80, W=200)
            for j, line in enumerate(str.split('\n')):
                y = y0 + j * dy
                imgArtText = DrawCustText(imgArtText, line, loc=(offsetX, y))
            offsetY += j * dy

    ls, nameList = [], []
    # ls.append(img),nameList.append('img')
    # ls.append(imgText),nameList.append('imgText')
    # ls.append(bImgText),nameList.append('bImgText')
    ls.append(imgArtText), nameList.append('')
    plotImagList(ls, nameList, gray=True, showticks=False)


def genMyPhotoStr(img):
    text = 'Hi,there!'
    str = getTextImgStr(text, H=30, W=180, font=cv2.FONT_HERSHEY_COMPLEX_SMALL | cv2.FONT_ITALIC)  # |cv2.FONT_ITALIC
    print(str)
    return

    H, W = getImgHW(img)
    img = resizeImg(img, W // 9, H // 9)
    # img = grayImg(img)
    img = cannyImg(img)
    infoImg(img)
    strPhoto = getGrayImageToStr(img)
    print(strPhoto)

    strPhoto = getBinaryTextImgStr(img, thresh=80, char='.', elseChar='#')
    print(strPhoto)


def main():
    img = loadImg(r'.\res\my.png')
    # generateArtText()

    # H,W = getImgHW(img)
    # img = resizeImg(img,W//4,H//4)
    infoImg(img)
    # getImageCharatStr(img)
    genMyPhotoStr(img)


if __name__ == '__main__':
    main()

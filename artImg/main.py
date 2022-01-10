# python3 steven, create art image use DNN training model
import cv2
import argparse

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#usgae:
#python .\\main.py -f ..\res\\Lenna.png -m .\\models\\eccv16\the_wave.t7
#python .\\main.py -f ..\res\\Lenna.png -m .\\models\\eccv16\the_wave.t7 -s -dst j.png
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def get_command_line_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('-f', type=str, required=True, help='src image')
    parser.add_argument('-m', type=str, required=True, help='DNN model path')
    # parser.add_argument('-s', default=False)
    parser.add_argument('-s', action="store_true", help='Want save the out image?')
    parser.add_argument('-dst', '--dst', type=str, help='Save path')

    # Parse and read arguments and assign them to variables if exists
    args, _ = parser.parse_known_args()

    srcImg = ''
    if args.f:
        srcImg = args.f

    model = ''
    if args.m:
        model = args.m

    save = False
    if args.s:
        save = True

    dst = '.'
    if args.dst:
        dst = args.dst

    return srcImg, model, save, dst


def showImg(img, name='Styled image'):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def writeImg(img, filePath):
    cv2.imwrite(filePath, img)


def loadImg(file, mode=cv2.IMREAD_COLOR):
    # mode = cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE cv2.IMREAD_UNCHANGED
    return cv2.imread(file, mode)

# model=r'./models/instance_norm/the_scream.t7'
# file=r'../res/Lenna.png'


def main():
    srcImg, model, save, dst = get_command_line_args()
    print(srcImg, model, save, dst)

    net = cv2.dnn.readNetFromTorch(model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    image = loadImg(srcImg)

    h, w = image.shape[0], image.shape[1]

    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
                                 (103.939, 116.779, 123.680),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out /= 255
    out = out.transpose(1, 2, 0)

    showImg(out)
    if save:
        out = out * 255
        writeImg(out, dst)


if __name__ == '__main__':
    main()

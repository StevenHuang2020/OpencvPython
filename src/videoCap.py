import cv2
# import numpy as np

gPath = './/res//'


def openCap():
    cap = cv2.VideoCapture(0)
    # getCap(cap)
    return cap


def cap():
    cap = openCap()
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here

        show = frame
        # show = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def capSave():
    cap = openCap()
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opncv 3.0
    # fourcc = cv2.cv.CV_FOURCC('X','V','I','D')

    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            # frame = cv2.flip(frame,0)#fanzhuan
            # write the flipped frame
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def getCap(cap):
    '''
    CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
    CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
    CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    CV_CAP_PROP_FPS Frame rate.
    CV_CAP_PROP_FOURCC 4-character code of codec.
    CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
    CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
    CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
    CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
    CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
    CV_CAP_PROP_HUE Hue of the image (only for cameras).
    CV_CAP_PROP_GAIN Gain of the image (only for cameras).
    CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
    CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
    CV_CAP_PROP_WHITE_BALANCE_U The U value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
    CV_CAP_PROP_WHITE_BALANCE_V The V value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
    CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
    CV_CAP_PROP_ISO_SPEED The ISO speed of the camera (note: only supported by DC1394 v 2.x backend currently)
    CV_CAP_PROP_BUFFERSIZE Amount of frames stored in internal buffer memory (note: only supported by DC1394 v 2.x backend currently)
    '''

    for i in range(19):
        print(i, cap.get(i))

    # ret = cap.set(3,320)
    # ret = cap.set(4,240)


def setCap(cap):
    pass


def playVideoFile(file):
    cap = cv2.VideoCapture(file)
    print(cap.isOpened())
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def capSetExposure():  # baoguang shezhi
    cap = openCap()
    proID = 16
    print('before exposure=', cap.get(proID))  # cv2.CV_CAP_PROP_EXPOSURE
    exposure = 0
    while(True):
        exposure += 1
        # exposure %=0

        exp = exposure / 10.0 - 10
        ret = cap.set(proID, exp)

        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        cv2.putText(frame, 'Exposure:%d,ac:%d' % (exp, cap.get(proID)), (20, 30), 3, 1.0, (255, 0, 0))
        show = frame
        # show = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # cap()
    # file = gPath+'FriendsS01E01.mkv' #"friends.mp4" #'H095428-33.avi'
    # playVideoFile(file)

    # capSave()
    capSetExposure()

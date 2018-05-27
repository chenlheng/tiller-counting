import cv2
import time


def video_to_picture(date, path):

    time_start = time.time()
    video = cv2.VideoCapture(path + date + '\\' + 'video.avi')
    f = 1

    if video.isOpened():
        rval, frame = video.read()
        print('video_to_picture in process')
    else:
        rval = False
        print('unable to open the video')

    while rval:
        cv2.imwrite('C:\\Users\Elden\Desktop\\' + date + '\\' + 'video_to_picture\\' + str(f) + '.jpg', frame)
        rval, frame = video.read()
        f = f + 1

    video.release()
    time_end = time.time()
    time = int(time_end - time_start)
    print('video_to_picture completed (time elapsed: ' + str(time) + 's)')


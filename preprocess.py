import cv2
import time
import os


input_path = '/home/lhchen/nas/MAISUI2/'
output_path = '/home/lhchen/nas/tiller_counting/data/raw/'

if __name__ == '__main__':

    time_start = time.time()
    files = os.listdir(input_path)
    for file in files:
        
        file_name = file[:-4]
        video = cv2.VideoCapture(input_path+file)
        f = 1
    
        if video.isOpened():
            rval, frame = video.read()
            print('video_to_picture in process')
        else:
            rval = False
            print('unable to open the video')
    
        while rval:
            cv2.imwrite(output_path+'%s-%i.jpg' % (file_name, f), frame)
            rval, frame = video.read()
            f = f + 1
    
        video.release()
        time_end = time.time()
        duration = int(time_end - time_start)
        print('%s -> %i frames in %is' % (file_name, f, duration))


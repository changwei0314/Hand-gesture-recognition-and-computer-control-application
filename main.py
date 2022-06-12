import cv2
import os
import time
import argparse
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
'''
GLOBAL VARIABLES
'''
# Layout/FrontEnd of image
IMAGEHEIGHT = 480  
IMAGEWIDTH = 640
ROIWIDTH = 256
LEFT = int(IMAGEWIDTH/2 - ROIWIDTH/2)
RIGHT = LEFT + ROIWIDTH
TOP = int(IMAGEHEIGHT/2 - ROIWIDTH/2)
BOTTOM = TOP + ROIWIDTH
SCOREBOXWIDTH = 320
BARCHARTLENGTH = SCOREBOXWIDTH-50
BARCHARTTHICKNESS = 15
BARCHARTGAP = 20
BARCHARTOFFSET = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX

gesture_dict= {0:'zero', 1:'one',2:'two' ,3:'three',4:'thumb_up', 5:'five',6:'six', 7:'seven',8:'ok', 9:'none'}
gesture_action={}

def parse_args():
    parser = argparse.ArgumentParser(
        description="data collection")
    parser.add_argument(
        '--clas',
        default=0,
        help='which class of the gestures',
    )
    return parser.parse_args()

if __name__ == '__main__':
    
    ARGS = parse_args()
    print("Activating your personal assistant:p...")

    cap= cv2.VideoCapture(0)

    while 1:
        label, image= cap.read()

        img=cv2.flip(img,1)
        cv2.putText(img, f"Press Enter to start your assistant.../nGesture in the rectangle to make the command :p", (80, 80), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
        cv2.imshow('Personal assistant',img) 
        key=cv2.waitKey(100)
        if  key ==13:
            break
        elif key==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    imgs=[]
    if key==13:
        while 1:
            label, image= cap.read()
            img=cv2.flip(img,1)
            cv2.putText(img, f"Press Enter to start your assistant.../nGesture in the rectangle to make the command :p", (80, 80), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
            cv2.imshow('Personal assistant',img) 
            img = cv2.bilateralFilter(img, 5, 50, 100)
            img = img[TOP:BOTTOM, LEFT:RIGHT]
            imgs.append(img)
            time.sleep(0.02)
            if len(imgs)==5:
                act= (imgs)
                imgs.clear()
                
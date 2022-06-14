import cv2
import os
import time
import argparse
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawingModule = mp.solutions.drawing_utils
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


# OpenCV image processing variables
BGSUBTHRESHOLD = 50
THRESHOLD = 50

# Gesture Mode variables
GESTUREMODE = False # Don't ever edit this!
GESTURES_RECORDED = [10,10,10,10,10,10,10,10,10,10]
gesture_dict= {0:'zero', 1:'one',2:'two' ,3:'three',4:'thumb_up', 5:'five',6:'six', 7:'seven',8:'ok', 9:'none', 10:'temp'}
# Gesture_index
ACTIONS_GESTURE_ENCODING = {'fist': 'Play/Unpause', 'five': 'Pause', 'none': 'Do Nothing', 'okay': 'Increase Volume', 'peace': 'Decrease Volume', 'rad': "Load Song", 'straight': "Stop", "thumbs":"NA"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="data collection")
    parser.add_argument(
        '--clas',
        default=0,
        help='which class of the gestures',
    )
    parser.add_argument(
        '--n',
        default=100,
        help='amount of images u wanna collect',
    )
    parser.add_argument(
        '--s',
        default=0,
        help='skeleton or not',
    )
    parser.add_argument(
        '--l',
        default=1000,
        help='amount limitation of the dataset of each class',
    )
    return parser.parse_args()

def show(img):
    cv2.imshow('temp',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    
    ARGS = parse_args()
    # Create a path for the data collection
    # img_label = create_path(WHERE, GESTURE)
    if int(ARGS.s):
        save_folder = os.path.join('data/training/skeleton', ARGS.clas)
    else:
        save_folder = os.path.join('data/training/no_skeleton', ARGS.clas)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # print(len(os.listdir()))

    print("Starting live video stream...")
    # bgModel = capture_background()

    # print(os.listdir(save_folder)[0])
    # img=cv2.imread("./data/training/skeleton/1/one_0.jpg",0)
    # show(img)

    # If a background has been captured
    n=len(os.listdir(save_folder))
    print(f'currently {n} images in this folder')
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(max_num_hands = 1,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:  
        while 1:
            label, image = cap.read()

            # Flip image
            image = cv2.flip(image,1)
            if int(ARGS.s):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.multi_hand_landmarks: 
                    for hand_landmarks in results.multi_hand_landmarks:
                        drawingModule.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                        drawingModule.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 1),
                                                        drawingModule.DrawingSpec(color = (250, 44, 250), thickness = 2, circle_radius = 1)) 
                        # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(image, f"Press Enter to start data collecting :)..", (80, 80), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(image, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
            cv2.imshow('Capturing data ',image) 
            key=cv2.waitKey(100)
            if  key ==13:
                break
            elif key==ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        
    if key==13:  
        with mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:  
            for i in range(int(ARGS.n)): 
                cur_n=len(os.listdir(save_folder))
                if cur_n==int(ARGS.l):
                    break
                # Capture image
                label, image = cap.read()

                # Flip image
                image = cv2.flip(image,1)
                
                if int(ARGS.s):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = hands.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            drawingModule.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                        drawingModule.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 1),
                                                        drawingModule.DrawingSpec(color = (250, 44, 250), thickness = 2, circle_radius = 1)) 
                        #     mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                cv2.putText(image, f"Data collecting please smile:)..{i}", (80, 80), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(image, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
                cv2.imshow('Capturing data ',image) 
                cv2.waitKey(10)
                # Applying smoothing filter that keeps edges sharp
                image = cv2.bilateralFilter(image, 5, 50, 100)

                # Selecting region of interest
                roi = image[TOP:BOTTOM, LEFT:RIGHT]

                time.sleep(0.5)
                cv2.imwrite(save_folder + f"/{gesture_dict[int(ARGS.clas)]}_{n+i}.jpg", roi)

        # Release the cap and close all windows if loop is broken
        cap.release()
        cv2.destroyAllWindows()
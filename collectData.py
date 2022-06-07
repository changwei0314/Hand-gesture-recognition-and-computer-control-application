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


# OpenCV image processing variables
BGSUBTHRESHOLD = 50
THRESHOLD = 50

# Gesture Mode variables
GESTUREMODE = False # Don't ever edit this!
GESTURES_RECORDED = [10,10,10,10,10,10,10,10,10,10]
SONG = 'The Beatles - I Saw Her Standing There'
ACTIONS_GESTURE_ENCODING = {'fist': 'Play/Unpause', 'five': 'Pause', 'none': 'Do Nothing', 'okay': 'Increase Volume', 'peace': 'Decrease Volume', 'rad': "Load Song", 'straight': "Stop", "thumbs":"NA"}

# Data Collection Mode variables
DATAMODE = False # Don't ever edit this!
WHERE = "train"
GESTURE = "okay"
NUMBERTOCAPTURE = 100

def parse_args():
    parser = argparse.ArgumentParser(
        description="data collection")
    parser.add_argument(
        '--clas',
        default='0',
        help='which class of the gestures',
    )
    parser.add_argument(
        '--n',
        default=100,
        help='amount of images u wanna collect',
    )
    parser.add_argument(
        '--s',
        default='0',
        help='skeleton or not',
    )
    return parser.parse_args()

# The controller/frontend that subtracts the background
def capture_background():
    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()

        if not ret:
            break
        
        image = cv2.flip(image,1)

        cv2.putText(image, "Press B to capture background.", (5, 50), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Press Q to quit.", (5, 80), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(image, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
        
        cv2.imshow('Capture Background',image)
        
        k = cv2.waitKey(5)
        
        # If key b is pressed
        if k == ord('b'):
            bgModel = cv2.createBackgroundSubtractorMOG2(0, BGSUBTHRESHOLD)
            # cap.release()
            cv2.destroyAllWindows()
            break
        
        # If key q is pressed
        elif k == ord('q'):
            bgModel = None
            cap.release()
            cv2.destroyAllWindows()
            break   

    return bgModel

# Remove the background from a new image
def remove_background(bgModel, image):
    fgmask = bgModel.apply(image, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(image, image, mask=eroded)
    return res

# Show the processed, thresholded image of hand in side image on right
def drawMask(image, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_image = 200*np.ones((IMAGEHEIGHT,ROIWIDTH+20,3),np.uint8)
    mask_image[10:266, 10:266] = mask
    cv2.putText(mask_image, "Mask",
                    (100, 290), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)
    return np.hstack((image, mask_image))

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

    # If a background has been captured
    n=len(os.listdir(save_folder))
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:  
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
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                cv2.putText(image, f"Data collecting please smile:)..{i}", (80, 80), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(image, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
                cv2.imshow('Capturing data ',image) 
                cv2.waitKey(10)
                # Applying smoothing filter that keeps edges sharp
                image = cv2.bilateralFilter(image, 5, 50, 100)
                
                # Remove background
                # no_background = remove_background(bgModel, image)
                # show(no_background)

                # Selecting region of interest
                roi = image[TOP:BOTTOM, LEFT:RIGHT]

                # Converting image to gray
                # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Blurring the image
                # blur = cv2.GaussianBlur(gray, (41, 41), 0)

                # Thresholding the image
                # ret, thresh = cv2.threshold(roi, THRESHOLD, 255, cv2.THRESH_BINARY)        
                
                # Draw new image with graphs
                # new_image = drawSideimage(HISTORIC_PREDICTIONS, image, 'Gesture Model', prediction_final)

                # Draw new dataimage with mask
                # new_image = drawMask(image,thresh)
                # new_image=image
                
                # If Datamode
                
                time.sleep(0.5)
                cv2.imwrite(save_folder + f"/{n+i}.jpg", roi)
                # cv2.putText(new_image, "Photos Captured:",(980,400), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)
                # cv2.putText(new_image, f"{i}/{NUMBERTOCAPTURE}",(1010,430), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)
        # Show the image
                # cv2.imshow('Gesture Jester', new_image)

                # key = cv2.waitKey(5)

                # # If q is pressed, quit the app
                # if key == ord('q'):
                #     break
        # Release the cap and close all windows if loop is broken
        cap.release()
        cv2.destroyAllWindows()
import cv2
import os
import time
import argparse
import numpy as np

'''
GLOBAL VARIABLES
'''
# Layout/FrontEnd of Frame
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
    return parser.parse_args()

# The controller/frontend that subtracts the background
def capture_background():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        frame = cv2.flip(frame,1)

        cv2.putText(frame, "Press B to capture background.", (5, 50), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press Q to quit.", (5, 80), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(frame, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
        
        cv2.imshow('Capture Background',frame)
        
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

# Remove the background from a new frame
def remove_background(bgModel, frame):
    fgmask = bgModel.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=eroded)
    return res

# Show the processed, thresholded image of hand in side frame on right
def drawMask(frame, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_frame = 200*np.ones((IMAGEHEIGHT,ROIWIDTH+20,3),np.uint8)
    mask_frame[10:266, 10:266] = mask
    cv2.putText(mask_frame, "Mask",
                    (100, 290), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)
    return np.hstack((frame, mask_frame))

def show(img):
    cv2.imshow('temp',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    
    ARGS = parse_args()
    # Create a path for the data collection
    # img_label = create_path(WHERE, GESTURE)

    save_folder = os.path.join('data/training/', ARGS.clas[0])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # print(len(os.listdir()))

    # use our model
    # light_img, _, _, _ = preprocess_image(f'data/test/images/{ARGS.light_image}', 2)

    print("Starting live video stream...")
    # bgModel = capture_background()

    # If a background has been captured
    n=len(os.listdir(save_folder))
    cap = cv2.VideoCapture(0)
    while 1:
        label, frame = cap.read()

        # Flip frame
        frame = cv2.flip(frame,1)
        cv2.putText(frame, f"Press Enter to start data collecting :)..", (80, 80), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
        cv2.imshow('Capturing data ',frame) 
        key=cv2.waitKey(100)
        if  key ==13:
            break
        elif key==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        
    if key==13:    
        for i in range(int(ARGS.n)): 
            # Capture frame
            label, frame = cap.read()

            # Flip frame
            frame = cv2.flip(frame,1)
            # show(frame)
            cv2.putText(frame, f"Data collecting please smile:)..{i}", (80, 80), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
            cv2.imshow('Capturing data ',frame) 
            cv2.waitKey(10)
            # Applying smoothing filter that keeps edges sharp
            frame = cv2.bilateralFilter(frame, 5, 50, 100)
            
            # Remove background
            # no_background = remove_background(bgModel, frame)
            # show(no_background)

            # Selecting region of interest
            roi = frame[TOP:BOTTOM, LEFT:RIGHT]

            # Converting image to gray
            # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Blurring the image
            # blur = cv2.GaussianBlur(gray, (41, 41), 0)

            # Thresholding the image
            # ret, thresh = cv2.threshold(roi, THRESHOLD, 255, cv2.THRESH_BINARY)        
            
            # Draw new frame with graphs
            # new_frame = drawSideFrame(HISTORIC_PREDICTIONS, frame, 'Gesture Model', prediction_final)

            # Draw new dataframe with mask
            # new_frame = drawMask(frame,thresh)
            # new_frame=frame
            
            # If Datamode
            
            time.sleep(0.25)
            cv2.imwrite(save_folder + f"/{n+i}.jpg", roi)
            # cv2.putText(new_frame, "Photos Captured:",(980,400), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)
            # cv2.putText(new_frame, f"{i}/{NUMBERTOCAPTURE}",(1010,430), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)
    # Show the frame
            # cv2.imshow('Gesture Jester', new_frame)

            # key = cv2.waitKey(5)

            # # If q is pressed, quit the app
            # if key == ord('q'):
            #     break
    # Release the cap and close all windows if loop is broken
        cap.release()
        cv2.destroyAllWindows()
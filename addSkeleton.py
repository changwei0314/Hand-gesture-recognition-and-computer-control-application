import cv2
import os
import mediapipe
from collectData import parse_args
import argparse

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

def addSkeleton(path):
    with handsModule.Hands(static_image_mode=True) as hands:

        image = cv2.imread(path)

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
 
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(image, handLandmarks, handsModule.HAND_CONNECTIONS,
                                        drawingModule.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 1),
                                        drawingModule.DrawingSpec(color = (250, 44, 250), thickness = 2, circle_radius = 1))
 
        #cv2.imshow('Test image', image)
        #plt.imshow(image)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image

def parse_args():
    parser = argparse.ArgumentParser(
        description="add skeleton")
    parser.add_argument(
        '-p',
        type = str,
        default=0,
        help='path of file or image'
    )
    return parser.parse_args()

if __name__ == "__main__":
    ARGS = parse_args()
    try:
        os.path.isdir(ARGS.p)
        flist = os.listdir(ARGS.p)
        for img in flist:
            img = addSkeleton(os.path.join(ARGS.p,img))
            cv2.imwrite(os.path.join(ARGS.p,img),img)
    except:
        try:
            img = addSkeleton(ARGS.p)
            cv2.imwrite(ARGS.p,img)

        except:
            print("not an image!!!")

import cv2
import os
import mediapipe

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

if __name__ == "__main__":
    Path = "data/training/no_skeleton"
    skeletonImgPath = "data/training/skeleton"
    for i in range (10):
        flist = os.listdir(Path+"/"+str(i))
        for imgPath in flist:
            skeletonImg = addSkeleton(Path+"/"+str(i)+"/"+imgPath)
            filename = imgPath.split("/")[-1]
            cv2.imwrite(skeletonImgPath+"/"+str(i)+"/"+filename,skeletonImg)
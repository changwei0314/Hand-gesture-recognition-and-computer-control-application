from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

from sklearn.svm import SVC
import cv2
import os
import mediapipe

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

def Addskeleton(path):
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


def Adaboost(dataPath="dataset",skeleton = 1):
    #Load data
    data = []
    label = []

    for i in range(10):
      path = dataPath+"/"+str(i)
      fptr = os.listdir(path)
      for filename in fptr:
        if skeleton:
          img = Addskeleton(path+"/"+filename)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
          img = cv2.imread(path+"/"+filename,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(30,30))
        img = np.asarray(img,dtype = float)
        img = img.reshape(1,-1)
        data.append(img)
        label.append(i)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3) # 70% training and 30% test
    
    svc = SVC(probability=True,kernel = 'linear')
    
    # Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=1)


    # Train Adaboost Classifer
    X_train = np.asarray(X_train)
    nsamples,nx,ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))
    X_test = np.asarray(X_test)
    nsamples,nx,ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))


    model = abc.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = model.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


    return model

if __name__ == "__main__":
    model = Adaboost()
    '''
    path = "evaluate/skeleton"
    data = []
    label = []
    for i in range(3):
        datapath = path+"/"+str(i)
        fptr = os.listdir(datapath)
        for filename in fptr:
            img = cv2.imread(datapath+"/"+filename,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(30,30),interpolation=cv2.INTER_AREA)
            img = np.asarray(img,dtype = float)
            img = img.reshape(1,-1)
            data.append(img)
            label.append(i)
    X_evaluate = np.asarray(data)
    nsamples,nx,ny = X_evaluate.shape
    X_evaluate = X_evaluate.reshape((nsamples,nx*ny))
    pred = model.predict(X_evaluate)
    print("Wild data Accuracy:",metrics.accuracy_score(label,pred))
    '''


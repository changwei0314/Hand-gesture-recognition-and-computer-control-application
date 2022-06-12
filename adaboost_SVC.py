from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import argparse

from sklearn.svm import SVC
import cv2
import os

from collectData import parse_args

def parse_args():
    parser = argparse.ArgumentParser(
        description="data collection")
    parser.add_argument(
        '--AddSkeleton',
        default=0,
        help='Add skeleton or not',
    )
    return parser.parse_args()


def Adaboost(dataPath="dataset",class_num=10,skeleton = 0):
    #Load data
    data = []
    label = []

    if skeleton:
      dataPath = dataPath+"/skeleton"
    else:
      dataPath = dataPath+"/no_skeleton"

    for i in range(class_num):
      path = dataPath+"/"+str(i)
      fptr = os.listdir(path)
      for filename in fptr:
        img = cv2.imread(path+"/"+filename,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(30,30))
        img = np.asarray(img,dtype = float)
        img = img.reshape(1,-1)
        data.append(img)
        label.append(i)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3,shuffle=True) # 70% training and 30% test
    
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

    print("Confusion_matrix:")
    print(metrics.confusion_matrix(y_test,y_pred))


    return model

if __name__ == "__main__":
    ARGS = parse_args()
    model = Adaboost("data/training",5,ARGS.AddSkeleton)
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


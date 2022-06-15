import numpy as np
import cv2
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

import argparse
from collectData import parse_args

class SAMME:
    """
    SAMME - multi-class AdaBoost algorithm
    @ref:   Zhu, Ji & Rosset, Saharon & Zou, Hui & Hastie, Trevor. (2006). Multi-class AdaBoost. Statistics and its
            interface. 2. 10.4310/SII.2009.v2.n3.a8.
    """

    def __init__(self, num_learner: int, num_cats: int):
        """
        Constructor
        :param num_learner: number of weak learners that will be boosted together
        :param num_cats: number of categories
        """

        if num_cats < 2:
            raise Exception("Param num_cat should be at least 2 but was {}".format(num_cats))

        self.num_learner = num_learner
        self.num_cats = num_cats
        self.entry_weights = None
        self.learner_weights = None
        self.sorted_learners = None

    def train(self, train_data: list,labels, learners: list):
        """
        Train the AdaBoost .
        The training data need to be in the format: [(X, label), ...]
        The learners need to be in the format: [obj1, obj2, ...]
        The learner object need to have: a predict method that can output the predicted class. obj.predict(X) -> cat: int
        :param train_data: training data
        :param learners: weak learners
        :return: void
        """

        print("\nStart training SAMME..")
        # initialize the weights for each data entry
        n, m = len(train_data), len(learners)
        self.entry_weights = np.full((n,), fill_value=1/n, dtype=np.float32)
        self.learner_weights = np.zeros((m,), dtype=np.float32)

        # sort the weak learners by error
        error = [0 for i in range(m)]
        for learner_idx, learner in enumerate(learners):
            learner.fit(train_data,labels)
            for idx,entry in enumerate(train_data):
                X = entry.reshape(1,-1)
                label = labels[idx]
                predicted_cat = learner.predict(X)
                if predicted_cat != label:
                    error[learner_idx] += 1
        self.sorted_learners = [l for l, e in sorted(zip(learners, error), key=lambda pair: pair[1])]

        # boost
        for learner_idx, learner in enumerate(self.sorted_learners):
            # compute weighted error
            is_wrong = np.zeros((n,))
            for entry_idx, entry in enumerate(train_data):
                X = entry.reshape(1,-1)
                label = labels[entry_idx]
                predicted_cat = learner.predict(X)
                if predicted_cat != label:
                    is_wrong[entry_idx] = 1
            weighted_learner_error = np.sum(is_wrong * self.entry_weights)/self.entry_weights.sum()

            # compute alpha, if the learner is not qualified, set to 0
            self.learner_weights[learner_idx] = max(0, np.log(1/(weighted_learner_error + 1e-6) - 1) + np.log(
                self.num_cats - 1))
            alpha_arr = np.full((n,), fill_value=self.learner_weights[learner_idx], dtype=np.float32)
            # update entry weights, prediction made by unqualified learner will not update the entry weights
            self.entry_weights = self.entry_weights * np.exp(alpha_arr * is_wrong)
            self.entry_weights = self.entry_weights/self.entry_weights.sum()

        # normalize the learner weights
        self.learner_weights = self.learner_weights/self.learner_weights.sum()
        print("Training completed.")

    def predict(self, X):
        """
        Predict using the boosted learner
        :param X:
        :return: predict class
        """

        pooled_prediction = np.zeros((self.num_cats,), dtype=np.float32)

        for learner_idx, learner in enumerate(self.sorted_learners):
            # encode the prediction in to balanced array
            predicted_cat = learner.predict(X.reshape(1,-1))
            prediction = np.full((self.num_cats,), fill_value=-1/(self.num_cats-1), dtype=np.float32)
            prediction[predicted_cat] = 1
            pooled_prediction += prediction*self.learner_weights[learner_idx]

        return np.argmax(pooled_prediction)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="train and test adaboost")
    parser.add_argument(
        '-n',
        type = int,
        default=30,
        help='Number of weakclassifier'
    )
    parser.add_argument(
        '-c',
        '--clsNum',
        type = int,
        default=10,
        help='Number of class.'
    )
    parser.add_argument(
        '-a',
        '--addSkeleton',
        default=1,
        help='Add skeleton or not.'
    )
    return parser.parse_args()



if __name__ == "__main__":
    ARGS = parse_args()
    num_learner = ARGS.n
    num_cats = ARGS.clsNum
    if(ARGS.addSkeleton):
        dataPath = "data/training/skeleton"
    else:
        dataPath = "data/training/no_skeleton"
    data = []
    label = []
    weak_learners = []
    


    #load data
    for i in range(num_cats):
      path = dataPath+"/"+str(i)
      fptr = os.listdir(path)
      for filename in fptr:
        img = cv2.imread(path+"/"+filename,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(30,30))
        img = np.asarray(img,dtype = float)
        img = img.reshape(1,-1)
        img = np.asarray(img)
        data.append(img)
        label.append(i)
    
    X_train, X_test,y_train,y_test = train_test_split(data,label, test_size=0.2,shuffle=True)
    
    train_data = np.asarray(X_train)
    nsamples,nx,ny = train_data.shape
    train_data = train_data.reshape((nsamples,nx*ny))
    test_data = np.asarray(X_test)
    nsamples,nx,ny = test_data.shape
    test_data = test_data.reshape((nsamples,nx*ny))
    #build weak learner
    for i in range(num_learner):
        clf = make_pipeline(StandardScaler(),SVC(gamma='auto'))
        weak_learners.append(clf)
    
    booster = SAMME(num_learner,num_cats)

    booster.train(train_data,y_train,weak_learners)

    y_pred = []
    for i in test_data:
        y_pred.append(booster.predict(i))
        
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    print("Confusion_matrix:")
    print(metrics.confusion_matrix(y_test,y_pred))
    print(metrics.classification_report(y_test,y_pred))

    



# Hand gesture recognition and computer control application
#### AI Final Project

## Introduction
>In this work, we train a CNN by using self-collected data with a skeleton on it. Then we gave the result that (1) our model outperformed the Adaboost baseline and (2) adding a skeleton on the dataset improves the model performance


## Collect Data
>1. Collecting data
>2. Adding skeleton on image
>`python addSkeleton.py [-h] [-p PATH]`
>The path can be the directory of an image or a folder


## Training
>1. Train Adaboost by running
>`python samme.py [-h] [-n NUM_LEARNER] [-c CLSNUM] [-a ADDSKELETON]`
  > * default datapath is data/training/skeleton/, -a can be switched to 0 to train on no_skeleton data
  > * n is the number of weak learners, and the default is 30
  > * c is the number of classes, and the default is 10, if it is set to less than 10(e.g., 5), AdaBoost will only train on the first five classes (0~4)
>2. Train CNN by running
>`train_cnn.ipynb`

# -*- coding: utf-8 -*-
from numpy import *
from csv import reader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import auc as auc3
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc as auc3
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import math
from allfiles import trainall, testall, trlabelall, telabelall
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot


def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label
if __name__ == '__main__':
    for index in range(len(trainall)):
        print(index)
        train = trainall[index]
        test = testall[index]
        trainlabel = trlabelall[index]
        trainlabel = trainlabel.T
        trainlabel = trainlabel.reshape(-1)
        testlabel = telabelall[index]
        testlabel = testlabel.T
        testlabel = testlabel.reshape(-1)

        rf= RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features=0.2)

        rf.fit(train, trainlabel)
        ae_y_pred_prob = rf.predict_proba(test)[:, 1]
        proba = transfer_label_from_prob(ae_y_pred_prob)
        np.savetxt('../output/score.txt', ae_y_pred_prob)
        np.savetxt('../output/probe.txt', proba)

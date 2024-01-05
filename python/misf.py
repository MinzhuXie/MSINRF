# -*- coding: utf-8 -*-

from numpy import *
import time
time1 = time.time()
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
def calculate_performace(test_num, pred_y, labels):  # pred_y = proba, labels = real_labels
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num

    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score = 0
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        f1_score = float(2 * tp) / ((2 * tp) + fp + fn)

    return acc, precision, sensitivity, specificity, MCC, f1_score


def transfer_array_format(data):  # data=X  , X= all the miRNA features, disease features
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])  # contains circRNA features ?
        formated_matrix2.append(val[1])  # contains disease features ?
    return np.array(formated_matrix1), np.array(formated_matrix2)

if __name__ == '__main__':
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    metric = np.zeros((1, 7))
    roc = []
    all_performance_DNN = []
    t = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
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
        #rf = svm.LinearSVC(C=1.0, max_iter=100)
        #rf=svm.SVC(C=0.8, kernel='rbf',degree=3, gamma=0.1, coef0=0.0, shrinking=True,probability=True)
        #rf=svm.SVC(C=0.8, kernel='rbf', gamma=6,probability=True)
        #rf=DecisionTreeClassifier()
        #rf=KNeighborsClassifier(n_neighbors=5)
        #rf=AdaBoostClassifier(n_estimators=100, learning_rate=0.8)
        rf.fit(train, trainlabel)
        #ae_y_pred_prob = rf.predict(test)

        ae_y_pred_prob = rf.predict_proba(test)[:, 1]
        np.savetxt('../Ucase/prob.txt', ae_y_pred_prob)
        proba = transfer_label_from_prob(ae_y_pred_prob)

        #metric = g_metrics(testlabel, ae_y_pred_prob)
        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(testlabel), proba,
                                                                                       testlabel)

        fpr, tpr, auc_thresholds = roc_curve(testlabel, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)


        precision1, recall, pr_threshods = precision_recall_curve(testlabel, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print ("AUTO-RF:", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score)
        all_performance_DNN.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score])
        t = t + 1  # AUC fold number

        pyplot.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))
        mean_tpr += interp(mean_fpr, fpr, tpr)  # one dimensional interpolation
        mean_tpr[0] = 0.0

        pyplot.xlabel('False positive rate, (1-Specificity)')
        pyplot.ylabel('True positive rate,(Sensitivity)')
        pyplot.title('Receiver Operating Characteristic curve: 5-Fold CV')
        pyplot.legend()

    mean_tpr /= len(trainall)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('*******AUTO-RF*****')
    print ('mean performance of rf using raw feature')
    print (np.mean(np.array(all_performance_DNN), axis=0))
    Mean_Result = []
    Mean_Result_std=[]
    Mean_Result = np.mean(np.array(all_performance_DNN), axis=0)
    Mean_Result_std = np.std(np.array(all_performance_DNN), axis=0)
    print('---' , Mean_Result_std)
    print ('---' * 20)
    print('Mean-Accuracy=', Mean_Result[0], '\n Mean-precision=', Mean_Result[1])
    print('Mean-Sensitivity=', Mean_Result[2], '\n Mean-Specificity=', Mean_Result[3])
    print('Mean-MCC=', Mean_Result[4], '\n' 'Mean-auc_score=', Mean_Result[5])
    print('Mean-Aupr-score=', Mean_Result[6], '\n' 'Mean_F1=', Mean_Result[7])
    print ('---' * 20)
    pyplot.plot(mean_fpr, mean_tpr, '--', linewidth=2.5, label='Mean ROC (AUC = %0.4f)' % Mean_Result[5])
    pyplot.legend()
    #np.savetxt('../model2/RWR_GATRFCDAx.txt', mean_fpr)
    #np.savetxt('../model2/RWR_GATRFCDAy.txt', mean_tpr)
    pyplot.show()
    end_time = time.time() - time1
    print('Embedding Learning Time: %.6f s' % end_time)

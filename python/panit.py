# -*- coding: utf-8 -*-
from numpy import *
from csv import reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from matplotlib import pyplot

#param=["KNN","AdaBoost","GATRFCDA","CNN","SVM"]
#param=["KNN","AdaBoost","MSINRF","CNN","SVM","RWR-RF","Raw-RF"]
param=["GATRFCDA","CD-LNLP","DMCCDA","SIMCCDA","NCPCDA","KATZHCDA"]
#param=["MSINRF"]
#param=["RWR_GATRFCDA","RWR_GATRFCDA-Negative sampling"]
#param=["RWR_GATRFCDA","CD-LNLP","NCPCDA","KATZHCDA","RWR"]
#auc=[0.9764]
auc=[0.9703,0.8016,0.9623,0.7284,0.9601,0.7953]
#auc=[0.9853,0.8016,0.9623,0.7284,0.9601,0.7953]
#auc=[0.8016,0.9623,0.9601,0.9764,0.7284,0.7953]
#auc=[0.8016,0.9623,0.7284,0.9601,0.9703,0.7953]
#auc=[0.7806,0.9401,0.9710,0.7923,0.7906]
for i in range(6):
      mean_fpr=np.loadtxt('../model2/'+param[i]+'x.txt')
      mean_tpr=np.loadtxt('../model2/'+param[i]+'y.txt')
      #mean_fpr=np.loadtxt('../model/'+param[i]+'_mean_fpr.txt')
      #mean_tpr=np.loadtxt('../model/'+param[i]+'_mean_tpr.txt')
      #auc_score = auc(mean_fpr, mean_tpr)
      pyplot.plot(mean_fpr, mean_tpr,linewidth=2.5, label='%s (AUC = %0.4f)' % (param[i], auc[i]))
      pyplot.xlabel('False positive rate')
      pyplot.ylabel('True positive rate')
      #pyplot.title('Receiver Operating Characteristic curve: 5-Fold CV')
      pyplot.legend()
pyplot.show()
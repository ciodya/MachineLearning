import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split # utils
from sklearn.metrics import mean_absolute_error, accuracy_score  # eval metric

# data processing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.pipeline import Pipeline

import os
#print(os.path.listdir("../input"))
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('normalizer', StandardScaler()), #Step1 - normalize data
    ('clf', SVC()) #step2 - classifier
])
#print(np.linspace(0.01,0.5,50))
#print(pipeline)


cv_grid = GridSearchCV(pipeline, cv=10,param_grid = {
    'clf__kernel' : ['linear','rbf','poly','sigmoid'],
    'clf__C' : np.linspace(0.01,0.5,50)
})

X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

list1 = [17,40,41,65,66,67,68,79,80,81,83,85,87]

xtrainin = np.zeros((600,112-len(list1)))
xtestin = np.zeros((1596,112-len(list1)))

k = 0

for j in range(112):
    if j not in list1:
        xtrainin[:,k] = X_train[:,j]
        xtestin[:,k] = X_test[:,j]
        k += 1

print(xtrainin.shape)
print(xtestin.shape)

#X_train, X_val, y_train, y_val = train_test_split(xtrainin, y_train, test_size=0.1, random_state=1)


cv_grid.fit(xtrainin, y_train)

#a_train, a_val, b_train, b_val = train_test_split(xtrainin, y_train, test_size=0.1, random_state=1)

#print(a_train.shape)

#cv_grid.fit(a_train, b_train)

#print(accuracy_score(b_val,cv_grid.predict(a_val)))

print(cv_grid.best_params_)

print(cv_grid.best_estimator_)

print(cv_grid.best_score_)

y_predict = cv_grid.predict(xtestin)


test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_predict
np.savetxt('SVMpugongyin.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")


'''

pipeline = Pipeline([
    ('normalizer', StandardScaler()), #Step1 - normalize data
    ('clf', SVC(C= 0.1)) #step2 - classifier
])

cv_grid = GridSearchCV(pipeline, param_grid = {
    'clf__kernel' : ['linear'],
})

cv_grid.fit(xtrainin, y_train)

print(cv_grid.best_params_)

print(cv_grid.best_estimator_)

print(cv_grid.best_score_)

y_predict = cv_grid.predict(xtestin)


test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_predict
np.savetxt('SVMcaonima.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
'''
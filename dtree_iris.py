# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:07:27 2019

@author: Arsalan Ashraf
"""


import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv("iris.data", names=names)


array = dataset.values
X = array[:,0:4]
Y = array[:,4]
t_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=t_size, random_state=seed)

clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
print(accuracy_score(Y_test, predictions))

i=0
while i<6:
    i=i+1
    clf = DecisionTreeClassifier(max_depth = i)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    print(accuracy_score(Y_test, predictions))

print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

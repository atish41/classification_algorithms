# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 14:14:35 2023

@author: ATISHKUMAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the dataset
dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\august\5th, 8th\Social_Network_Ads.csv')

#set the target varible and i.v
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#aplit the dataset into training and testing 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20,random_state=0)

'''#we apply featurre scaling technique here 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)'''

#we apply decision tree algorithm to find accuracy gain
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini',splitter='random',max_depth=5,min_samples_split=2,min_samples_leaf=1)
classifier.fit(x_train, y_train)

#test the results
y_pred=classifier.predict(x_test)

#confusion matrix for evaulate 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm

#accuracy score
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
ac

bias =classifier.score(x_train, y_train)
bias

verience=classifier.score(x_test,y_test)
verience

#classification report
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr



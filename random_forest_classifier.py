# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:16:15 2023

@author: ATISHKUMAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the dataset
dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\august\5th, 8th\Social_Network_Ads.csv')

#split the dataset into i.v and d.v
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#split into trainning and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20,random_state=0)

'''#we apply feature scaling in this but feature scaling is not required beacoz its traa algorithm
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)'''

#we apply random forest algorithm
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=110,criterion='gini',max_depth=5)
classifier.fit(x_train, y_train)

#check the results
y_pred=classifier.predict(x_test)

#confusiion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

#accuracy score
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
ac

bias =classifier.score(x_train, y_train)
bias

varaience=classifier.score(x_test, y_test)
varaience

#classification report
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr



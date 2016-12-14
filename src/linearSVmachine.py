#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:14:21 2016

@author: edoardoghini
"""
from sklearn import  datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import colors, pyplot
import math
import numpy as np
    
    
    
#parameters definition
split_rate_test=0.5
split_test_over_validation= 0.6
cmap_light = colors.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])



def plot_boundaries(test_matrix,test_labels,clf,cmap,title):
    X= test_matrix
    h=0.1
    x_min, x_max= X[:,0].min() -1, X[:,0].max() +1
    y_min, y_max= X[:,1].min() -1, X[:,1].max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z= clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z= Z.reshape(xx.shape)
    pyplot.figure()
    pyplot.pcolormesh(xx, yy, Z, cmap=cmap_light)
    pyplot.scatter(X[:,0],X[:,1],c=test_labels,cmap=cmap)
    pyplot.title(title)
    
    
    
    
#loading data sets
iris = datasets.load_iris()
X=iris.data
y= iris.target
#take the first and second columns of data
X= X[:,:2]
#splitting into train, validation and test
X_train_not_std, X_mixed_not_std, y_train, y_mixed= train_test_split(X, y,  test_size=split_rate_test, random_state=20)
X_validation_not_std, X_test_not_std, y_validation, y_test= train_test_split( X_mixed_not_std, y_mixed, test_size= split_test_over_validation, random_state=10)
'''
#standardizing dataset  ## FOR NOW THE STANDARDIZE OF DATAS SEEMS MISLEADING BECAUSE WE NEED A PROPER REPRESENTATION OF SUPPORT VECTORS THAT OTHERWISE COULD BECAME A WRONG APPROXIMATION OF THE INITIAL DATASET
X_scaler = preprocessing.StandardScaler()
X_train = X_scaler.fit_transform(X_train_not_std)
X_validation = X_scaler.fit_transform(X_validation_not_std) 
X_test = X_scaler.transform(X_test_not_std)
'''
#train a linear svm for different values of C
scores= []
for i in range(-3,4):
    
    C_param= math.pow(10,i)
    svm = SVC(kernel='linear', C=C_param, random_state=0)
    svm.fit(X_train_not_std, y_train)

    #store statistics & plotting boundaries
    elem= [i , svm.score(X_validation_not_std, y_validation)]
    scores.append(elem)
    plot_boundaries(X_validation_not_std,y_validation,svm,cmap_light, title=C_param)
    
#score statistics
x_axis=[]
y_axis=[]
best=[]
for e in scores:
    c_value , score = e
    x_axis.append(c_value)
    y_axis.append(score)
    if not len(best) :
        best = e
    elif score > best[1]:
        best = e
#plot statistics    
pyplot.figure()
pyplot.plot(x_axis,y_axis)
pyplot.title("overall score within C param variation")

#best performer svm on test data
C_param= best[0]
svm = SVC(kernel='linear', C=C_param, random_state=0)
svm.fit(X_train_not_std, y_train)

#plotting boundaries
plot_boundaries(X_test_not_std,y_test,svm,cmap_light, title="best SVM with C parameter " + str(C_param))
    
    
    

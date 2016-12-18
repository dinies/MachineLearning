#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:36:42 2016

@author: edoardoghini
"""

from sklearn import  datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import colors, pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


import math
import numpy as np
    
    
    
#parameters definition
split_rate_test=0.5
split_test_over_validation= 0.6
split_rate_merged= 0.3
cmap_light = colors.ListedColormap(['#C71585', '#FF7F50', '#40E0D0'])



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
    svm = SVC(kernel='rbf', C=C_param, random_state=0)
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
C_param= math.pow(10,best[0])
svm = SVC(kernel='rbf', C=C_param, random_state=0)
svm.fit(X_train_not_std, y_train)

#plotting boundaries
plot_boundaries(X_test_not_std,y_test,svm,cmap_light, title="best SVM with C parameter " + str(C_param))

#grid search with both gamma and C parameter variations

X_train_merged, X_test_merged, y_train_merged, y_test_merged= train_test_split(X, y,  test_size=split_rate_merged, random_state=20)

C_range = np.logspace(-3, 6, 10)
gamma_range = np.logspace(-6, 3, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_train_merged, y_train_merged)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#best performer from the grid search svm on test data
best_C= grid.best_params_['C']
best_gamma= grid.best_params_['gamma']

svm = SVC(kernel='rbf', C=best_C, gamma=best_gamma,  random_state=0)
svm.fit(X_train_merged, y_train_merged)

score=svm.score(X_test_merged,y_test_merged)
print("best score"+ str(score))
plot_boundaries(X_test_merged,y_test_merged,svm,cmap_light, title="best SVM with C "+str(best_C)+ "and gamma "+str(best_gamma)+" parameters from grid search ")





    

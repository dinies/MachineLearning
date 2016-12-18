#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:15:39 2016

@author: edoardoghini
"""
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import colors, pyplot




def plot_boundaries(test_matrix,test_labels,neighbour_classifier,cmap,title):
    X= test_matrix
    h=0.1
    x_min, x_max= X[:,0].min() -1, X[:,0].max() +1
    y_min, y_max= X[:,1].min() -1, X[:,1].max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z= neighbour_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z= Z.reshape(xx.shape)
    pyplot.figure()
    pyplot.pcolormesh(xx, yy, Z, cmap=cmap_light)
    pyplot.scatter(X[:,0],X[:,1],c=test_labels,cmap=cmap)
    pyplot.title(title)


    
#parameters definition
split_rate=0.4
pca_param=2
num_neighbors=40
cmap_light = colors.ListedColormap(['#C71585', '#FF7F50', '#40E0D0'])

#loading data sets
iris = datasets.load_iris()
X=iris.data
y= iris.target
#splitting into train and test
X_train_not_std, X_test_not_std, y_train, y_test= train_test_split(X, y,test_size=split_rate,random_state=80)
#standardizing dataset 
X_scaler = preprocessing.StandardScaler()
X_train = X_scaler.fit_transform(X_train_not_std)
X_test = X_scaler.transform(X_test_not_std)
#pca reduction
X_train_pca=PCA(pca_param).fit_transform(X_train)
X_test_pca=PCA(pca_param).fit_transform(X_test)

#plot of training data
pyplot.figure()
pyplot.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train,cmap=cmap_light)
pyplot.title("Iris dataset")
#iteration of different values for neighbors
scores=[]
for i in range(1,11):
    
    #training the model
    clf= neighbors.KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train_pca, y_train)
    #store statistics & plotting boundaries
    elem= [ i,  clf.score(X_test_pca, y_test)  ]
    scores.append(elem)
    plot_boundaries(X_test_pca,y_test,clf,cmap_light,title="number of neighbors "+ str(i))
#score statistics

x_axis=[]
y_axis=[]
for e in scores:
    n_neigh, score = e
    x_axis.append(n_neigh)
    y_axis.append(score)

pyplot.figure()
pyplot.plot(x_axis,y_axis)
pyplot.title("score statiistics")

#weight functions showdown uniform vs distance
#training
clf= neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance')
clf.fit(X_train_pca, y_train)
#boundaries
plot_boundaries(X_test_pca,y_test,clf,cmap_light,title="weight function type: distance")
#training (explicit parameter for weights)
clf= neighbors.KNeighborsClassifier(n_neighbors=3,weights='uniform')
clf.fit(X_train_pca, y_train)
#boundaries
plot_boundaries(X_test_pca,y_test,clf,cmap_light,title="weight function type: uniform")

for alfa in [0.1,10,100,1000]:
    

    #square_distance_gaussian weights
    def square_gaussian_distance(distance):
        return np.exp(  - 0.1 * alfa * distance** 2)
        
    #training
    clf= neighbors.KNeighborsClassifier(n_neighbors=3,weights=square_gaussian_distance)
    clf.fit(X_train_pca, y_train)
    #boundaries
    print ("alfa",alfa)
    print('score',clf.score(X_test_pca, y_test))
    plot_boundaries(X_test_pca,y_test,clf,cmap_light, title=" gaussian weight with alfa "+str(alfa))           #PERCHE NON CAMBIA NULLA DA UN PUNTO DI VISTA DEGLI SCORE E DEI BOUNDARIES AL VARIARE DEL PARAMETRO ALFA  ??

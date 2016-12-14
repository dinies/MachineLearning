#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:46:33 2016

@author: edoardoghini
"""
import numpy as np
import sklearn.preprocessing as prepro
import sklearn.linear_model as linmod
import matplotlib.pyplot as plt
import math

class polinomialexpander(object):
    
    def __init__(self):
        self.X_train= np.load('regressionDataset/regression_Xtrain.npy')
        self.y_train= np.load('regressionDataset/regression_ytrain.npy')
        self.X_test= np.load('regressionDataset/regression_Xtest.npy')
        self.y_test= np.load('regressionDataset/regression_ytest.npy')
        
    def do(self):
        
        #plotting data sets
        plt.figure()
        plt.scatter(self.X_train,self.y_train)
        plt.show()
        plt.figure()
        plt.scatter(self.X_test,self.y_test)
        plt.show()
        
        #standardizing datasets
        X_scaler = prepro.StandardScaler()
        X_train = X_scaler.fit_transform(self.X_train.reshape(-1,1))
        X_test = X_scaler.transform(self.X_test.reshape(-1,1))
        #y_scaler = prepro.StandardScaler()
        y_train = self.y_train 
        #y_scaler.fit_transform(self.y_train[:, None])[:, 0]
        y_test = self.y_test
        #y_scaler.transform(self.y_test[:, None])[:, 0]
        '''
        #plotting data after standardization
        plt.figure()
        plt.scatter(X_train,y_train)
        plt.show()
        plt.figure()
        plt.scatter(X_test, y_test)
        plt.show()
        
        #linear fitting
        lr= linmod.LinearRegression()
        lr.fit(X_train.reshape(-1,1),y_train)
        
        #plotting results
        plt.figure()
        plt.plot(X_test, lr.predict(X_test.reshape(-1,1)),label="Model")
        plt.scatter(X_test, y_test)
        plt.show()
        
        #mean square error computation
        predicted= lr.predict(X_test.reshape(-1,1))
        inc=0
        for index in range(len(predicted)):
            inc += math.pow(y_test[index]-predicted[index],2)
        mean_square_error = inc/len(predicted)
        print ('mse',mean_square_error)
       
        #polinomial expansion
        poly=prepro.PolynomialFeatures(3)
        X_poli= poly.fit_transform(X_train,y_train)
        
        #repetition of previous steps
        #linear fitting
        lr= linmod.LinearRegression()
        lr.fit(X_poli,y_train)
        #choosing boundaries of the plot
        x_range = np.linspace(-2,2, 100)
        predicted = lr.predict(poly.fit_transform(x_range.reshape(-1,1)))
        #plotting results
        plt.figure()
        plt.plot(x_range.reshape(-1,1), predicted)
        plt.scatter(X_test.reshape(-1,1),y_test, c='r')
        plt.show()
        
        #mean square error computation with polinomial expansion of xtest
        
        predicted= lr.predict(X_test.reshape(-1,1))
        inc=0
        for index in range(len(predicted)):
            inc += math.pow(y_test[index]-predicted[index],2)
        mean_square_error = inc/len(predicted)
        print ('mse',mean_square_error)
        '''
        #ranged polinomial expansion
        x_axis=[]
        y_axis=[]
        models={}
        for i in range(1,10):
            #polinomial expansion
            poly=prepro.PolynomialFeatures(i)
            X_poli_train= poly.fit_transform(X_train,y_train)
            #linear fitting
            lr= linmod.LinearRegression()
            lr.fit(X_poli_train,y_train)
            #mean square error computation with X_test poly expansion
            X_poli_test=poly.fit_transform(X_test,y_test)
            predicted= lr.predict(X_poli_test)
            inc=0
            for index in range(len(predicted)):
                inc += math.pow(y_test[index]-predicted[index],2)
            mse = inc/len(predicted)
            
            x_axis.append(i)
            y_axis.append(mse)
            models[i]= {'X':X_poli_train, 'mse':mse}
            #plot for the particular polinomial
            x_range = np.linspace(-2,2, 100)
            predicted = lr.predict(poly.fit_transform(x_range.reshape(-1,1)))
            #plotting results
            plt.figure()
            plt.plot(x_range.reshape(-1,1), predicted)
            plt.scatter(X_test.reshape(-1,1),y_test, c='r')
            plt.show()
            plt.title("polinomial of order "+ str(i)+" with mse "+ str(mse) )
        #plot mse for polinomial
        plt.figure()
        plt.plot(x_axis,y_axis)
        plt.title("various mse with grade variation of polinomial")
        #check for the model with minimum mse
        min_mse_index=1
        for key in models.keys():
            if models[min_mse_index]['mse'] > models[key]['mse'] :
                min_mse_index= key
        poly= prepro.PolynomialFeatures(min_mse_index)
        lr= linmod.LinearRegression()
        lr.fit(models[min_mse_index]['X'],y_train)
        #choosing boundaries of the plot
        x_range = np.linspace(-2,2, 100)
        print(min_mse_index)
        predicted = lr.predict(poly.fit_transform(x_range.reshape(-1,1)))
        #plotting results
        plt.figure()
        plt.plot(x_range.reshape(-1,1), predicted)
        plt.scatter(X_test.reshape(-1,1),y_test, c='r')
        plt.show()
        plt.title("mean square error"+ str(float(models[min_mse_index]['mse']))+ "with polynomial expansion of order"+ str(min_mse_index))
                    
            
            
        
        
        
x= polinomialexpander()
x.do()        
                
                        
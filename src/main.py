#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:40:44 2016

@author: edoardoghini
"""

'''
b= bysclf.bysclf()
b.do_plot()
'''
import polinomialexpander as p

bum= p.polinomialexpander()
z=bum.X_test
w=bum.y_test
x=bum.X_train
v=bum.y_train
bum.do()
print ('hello')



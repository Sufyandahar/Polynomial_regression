#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 06:33:54 2020

@author: sufiyan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# linearRegression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_Poly = poly_reg.fit_transform(X)
#poly_reg.fit(X_Poly, y) 
lin_reg2 = LinearRegression()
lin_reg2.fit(X_Poly, y)


# visiualizing the data of linear regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# visiualizing the data of polynomial regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


######################### Predicted new value with (linear regression) ##############################
#lin_reg.predict([[input("Enter your position level :")]])


######################### Predicted new value with (Polynomial regression) ##############################
lin_reg2.predict(poly_reg.fit_transform([[input("Enter your Position level :")]]))
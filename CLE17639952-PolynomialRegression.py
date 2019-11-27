#List of imports used throughout the application 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import numpy.linalg as linalg

data = 'CMP3751M_ML_Assignment 1_Task1_pol_regression.csv'
data = pd.read_csv(data)

#Defined degrees used for polynormal feature expansion 
degrees = [0, 1, 2, 3, 5, 10]

RMSETrain = []
RMSETest = []

#Visual plotting for X and Y  
x = data['x']
y = data['y']
x = np.sort(x, axis=0)
y = np.sort(y, axis=0)
plt.figure()
plt.plot(x, y, 'bo')
plt.xlabel('X')
plt.ylabel('Y')

#Polynormal feature expansion, creates an array the size of x,
# for each data value times power it based on the degree. 
def Designmatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i))
    return X

#The least squares solution calculation, calling upon the 
#polynormal feature expansion then finding the respected weights
#by transposing X, multiplying it by itself. Then using the linalg solve 
#function to avoid direct matrix inversions, multiplying X transposed against 
#Y and calculate together to find the least squared values.
def pol_regression(x, y, degree):
    X = Designmatrix(x, degree)
    XX = X.transpose().dot(X)
    print(XX)
    w = np.linalg.solve(XX, X.transpose().dot(y))
    return w

#The evaluation polynormal regression function calculates the root 
#mean squared error and returns the value.
def eval_pol_regression(parameters, x, y): 
    rmse = np.sqrt(((x.dot(parameters) - y) ** 2).mean())
    return rmse

#Section 1.3 Evaluation
#The evaluation function calculates and plots all the necessary 
#data to evaluate the calculated training set to the test data.
#This is done by splitting x and y into a 70, 30 spilt. 
#Previous functions are called to calculate the necessary
#values to evaluate the polynormal regression. 
def evaluation():
    index = 0 
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, shuffle=True)
    for d in degrees[1:]:

        x_train = Designmatrix(xtrain ,d)
        x_test = Designmatrix(xtest ,d)
        w = pol_regression(xtrain, ytrain, d)
        RMSETrain.append( eval_pol_regression(w, x_train, ytrain) )
        RMSETest.append( eval_pol_regression(w, x_test, ytest) ) 
        index+=1
    plt.semilogy(degrees[1:], RMSETrain)
    plt.semilogy(degrees[1:], RMSETest)
    plt.legend(('Train','Test'), loc = 'upper left')
    plt.show()

#Section 1.2 Implementation of Polynomial Regression
#The main function goes through the list of degrees, 
# and calls upon the previous functions to calculate the 
#least square solution and plots the result to be 
#visually displayed. 
def main():

    for d in degrees:
        if d is 0: plt.plot([-5, 5],[y.mean(), y.mean()])
        else:
            w = pol_regression(x, y, d)
            X = Designmatrix(x ,d)
            X = X.dot(w)
            print('Degree : ' + str(d) + ' : ' + str(w))
            plt.plot(x, X)
    plt.xlim((-6, 6))
    plt.legend(('ground truth', '$x$', '$x^2$', '$x^3$', '$x^5$', '$x^10$'), loc = 'lower right')
    #plt.legend(('ground truth','$x^10$'), loc = 'lower right')

    plt.show()
    evaluation()

main()

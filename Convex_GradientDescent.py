import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib import expand_dims
from math import sqrt

x = [] 


x1 = np.expand_dims(np.array([1,1]), -1) #Initial Value
x_stationary = np.array([-1.5, -0.5]) #Stationary Point
hessianMatrix = np.array([[6,6],[6,16]]) #Hessian Mtarix which is constant for this function
x.append(x1)

def f(a):  #function
    fValue =  3*a[0]**2 + 12*a[0] + 8*a[1]**2 + 8*a[1] + 6*a[1]*a[0]
    return fValue

def gradF(a): #Gradient Function
    fGradValue = np.array([6*a[0] + 12 + 6*a[1] , 16*a[1] + 8 + 6*a[0]])
    return fGradValue

def errorFunc(x): #Error Function
    error = sqrt((x[1] - x_stationary[1])**2 + (x[0] - x_stationary[0])**2)
    return error

def stepDirec(fGradValue): #Direction 
    Direc = np.dot(hessianMatrix, fGradValue) * -1
    return Direc

def stepValue(p, a): #Calculation of step value (alpha)
    alpha = -1 * (6 * p[0] * (a[0] + a[1]) + 2 * p[1] * (8*a[1] + 3*a[0]) + 4 * (3*p[0] + 2 * p[1])) / (6* p[0]**2 + 16*p[1]**2 + 12* p[0]*p[1])
    return alpha[0]

while True:
    x_curr = x[-1]
    fGradValue = gradF(x_curr) 
    p_curr = stepDirec(fGradValue)
    step_curr = stepValue(p_curr, x_curr)
    x_next = (x_curr + step_curr * p_curr)
    if errorFunc(x_next) < 0.5:
        break
    else:
        if (x_next == x_curr).all() :
            break
        x.append(x_next)
        print(x_next)





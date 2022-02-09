import numpy as np
from math import sqrt, cos, sin, pow
import math


x1 = np.array([0,0]) #Initial Value
x2 = np.array([10,-5]) #Initial Value

c1 = 0.3
beta = 0.8

def f(a):  #function
    fValue =  pow(a[0], 2) - 10 * a[1] * cos(0.2 * np.pi * a[0]) + pow(a[1], 2) - 15 * a[0]* cos(0.4* np.pi * a[1])
    return fValue


def gradF(a): #Gradient Function
    fGradValue = np.array([6.28319 *a[1]* math.sin(0.628319*a[0]) + 2*a[0] - 15 * math.cos(1.25664 *a[1]), 
    18.8496 * a[0] * math.sin(1.25664 *a[1]) - 10* math.cos(0.628319 * a[0]) + 2 *a[1]])
    return fGradValue

def hessianMatrix(a): #Hessian Matrix
    hessian = np.array([[3.9478 * a[1] * cos(0.6283 * a[0]) + 2, 6.2831 * sin(0.6283 * a[0]) + 18.8496 * sin(1.2566 * a[1])],
        [6.2831 * sin(0.6283 * a[0]) + 18.8496 * sin(1.2566 * a[1]), 23.6871 * a[0] * cos(1.2566 * a[1]) + 2]])
    return hessian

def stepDirec(fGradValue, a): #Direction 
    Direc = np.dot(hessianMatrix(a), fGradValue) * -1
    return Direc

def stepValueArmijo(alpha, pVal, xVal, fGradVal): #Calculation of Step Value using Armijo Rule
    numOfMul = 0
    initAlpha =  alpha
    while f(xVal + alpha * pVal) > f(xVal) + c1 * alpha * np.dot(fGradVal.T, pVal): #under this condition
        alpha *= beta #update alpha
        numOfMul += 1
        if alpha / initAlpha < 0.1: 
            alpha = initAlpha / 10
            break
    # print(numOfMul)
    return alpha

fStationary = -75.6
initAlpha = 0.4

iters = 100000


xVals = [x1]
step_curr = initAlpha
prevF = 0

for i in range(iters): #first point (0,0)
    x_curr = xVals[-1]
    fGradValue = gradF(x_curr)
    p_curr = stepDirec(fGradValue, x_curr)

    if i == 0:
        x_next = (x_curr + step_curr * p_curr)
        prevF = f(x_curr)
    elif i == 1:
        step_curr = 2 * (f(x_curr) - prevF) / np.dot(fGradValue, p_curr) #start armijo rule after one iteration
        x_next = (x_curr + step_curr * p_curr)
    else:
        step_curr = stepValueArmijo(step_curr, p_curr, x_curr, fGradValue)
        x_next = (x_curr + step_curr * p_curr)
    
  
    xVals.append(x_next)
    # print(x_next, step_curr)

print('for x =', x1, ', optimized f value is: ', f(x_next))


xVals = [x2]
step_curr = 0.01
for _ in range(iters):  #second point (10,-5)
    x_curr = xVals[-1]
    fGradValue = gradF(x_curr)
    p_curr = stepDirec(fGradValue, x_curr)
    
    if i == 0:
        x_next = (x_curr + step_curr * p_curr)
        prevF = f(x_curr)
    elif i == 1:
        step_curr = 2 * (f(x_curr) - prevF) / np.dot(fGradValue, p_curr) #start armijo rule after one iteration
        x_next = (x_curr + step_curr * p_curr)
    else:
        step_curr = stepValueArmijo(step_curr, p_curr, x_curr, fGradValue)
        x_next = (x_curr + step_curr * p_curr)
  
    xVals.append(x_next)
    # print(x_next, step_curr)

print('for x =', x2, ', optimized f value is: ', f(x_next))

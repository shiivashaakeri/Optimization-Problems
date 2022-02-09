import numpy as np
from math import sqrt
import math


x = []
a1 = np.array([10,-5]) #Initial Value
a2 = np.array([-1,-2]) #Initial Value


hessianMatrix = np.array([[6,6],[6,16]]) #Hessian Mtarix which is constant for this function


def f(a):  #function
    fValue =  a[0] **2 - 10*a[1] * math.cos(0.2* math.pi *a[0]) + a[1]**2 - 15*a[0]* math.cos(0.4* math.pi * a[1])
    return fValue


def gradF(a): #Gradient Function
    fGradValue = np.array([6.28319 *a[1]* math.sin(0.628319*a[0]) + 2*a[0] - 15 * math.cos(1.25664 *a[1]), 
    18.8496 * a[0] * math.sin(1.25664 *a[1]) - 10* math.cos(0.628319 * a[0]) + 2 *a[1]])
    return fGradValue


iters = 100 #number of iterations
fStationary = -75.6 #function value in local min
alpha = 0.01 #initial alpha

x.append(a1)
for _ in range(iters): #first point
    x_curr = x[-1]
    fGradValue = gradF(x_curr) # evaluate the gradient of initial point
    x_next = (x_curr - alpha * fGradValue) #calculate next step point
  
    if (x_next == x_curr).all() :
        break
    x.append(x_next)

print('for x =', a1, ', optimized f value is: ', f(x_next))

x.append(a2)
for _ in range(iters): #second point
    x_curr = x[-1]
    fGradValue = gradF(x_curr)
    x_next = (x_curr - alpha * fGradValue)
  
    if (x_next == x_curr).all() :
        break
    x.append(x_next)

print('for x =', a2, ', optimized f value is: ', f(x_next))
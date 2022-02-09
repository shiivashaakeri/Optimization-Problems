from math import sqrt, cos, sin, pow
import numpy as np
 

def f(x): #Function Definition
	return pow(x[0], 2) - 10 * x[1] * cos(0.2 * np.pi * x[0]) + pow(x[1], 2) - 15 * x[0]* cos(0.4* np.pi * x[1])
 

def simulated_annealing(iters, stepVal, temp): #Simulated Annealing
	
	x = (0, 0) #initial point

	# print(bestPoint)
    
	funcVal = f(x) # evaluate the initial point

	x_curr, funcVal_curr = x, funcVal

	for _ in range(iters):
		x_next = (x_curr[0] + np.random.randn(len(np.asarray([[-15.0, 15.0]]))) * stepVal, x_curr[1] + np.random.randn(len(np.asarray([[-15.0, 15.0]]))) * stepVal)
		funcVal_next = f(x_next)

		if funcVal_next < funcVal:

			x, funcVal = x_next, funcVal_next
		diff = funcVal_next - funcVal_curr 
		t = temp / float(_ + 1) 	# calculate temperature 
		shouldAccept = np.exp(-diff / t)
		if diff < 0 or np.random.rand() < shouldAccept:
			x_curr, funcVal_curr = x_next, funcVal_next
	return x, funcVal
 

print( simulated_annealing( 10000, 2, 15) )
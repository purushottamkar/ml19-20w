'''
	Package: cs771
	Module: optLib
	Author: Puru
	Institution: CSE, IIT Kanpur
	License: GNU GPL v3.0
	
	Give skeletal support for implementing various optimization techniques like gradient and coordinate methods
'''

import numpy as np
from matplotlib import pyplot as plt
import time as t
import random

# The word "oracle", as used below, refers for a function that does a certain job to our satisfaction.
# For example, our step length oracle returns a step length value whenever we ask it, the gradient
# oracle always returns a direction either by calculating the gradient, or the (mini-batch) stochastic gradient

# Get functions that offer various step length sequences. The constant step length scheme is only suitable for
# "superbly nice" functions that satisfy two properties known as "strong convexity" and "strong smoothness". 
# The quadratic scheme decreases step lengths gradually and is a good choice for several cases, whether "nice" or not.
# The linear scheme decreases step lengths fairly rapidly and is suitable for functions with intermediate "niceness".
# Examples include strongly convex functions that are not strongly smooth (the CSVM objective is one such example) or
# strongly smooth functions that are not strongly convex.
def stepLengthGenerator( mode, eta ):
	if mode == "constant":
		return lambda t: eta
	elif mode == "linear":
		return lambda t: eta/(t+1)
	elif mode == "quadratic":
		return lambda t: eta/np.sqrt(t+1)

# Given a gradient oracle and a steplength oracle, implement the gradient descent method
# The method returns the final model and the objective value acheived by the intermediate models
# This can be used to implement (mini-batch) SGD as well by simply modifying the gradient oracle
# The postGradFunc can be used to implement projections or thresholding operations
def doGD( gradFunc, stepFunc, objFunc, init, horizon = 10, doModelAveraging = False, postGradFunc = None ):
	objValSeries = []
	timeSeries = []
	totTime = 0
	theta = init
	cumulative = init
	
	for it in range( horizon ):
		# Start a stopwatch to calculate how much time we are spending
		tic = t.perf_counter()
		delta = gradFunc( theta, it + 1 )
		theta = theta - stepFunc( it + 1 ) * delta
		if postGradFunc is not None:
			theta = postGradFunc( theta , it + 1 )
		# If we are going to do model averaging, just keep adding the models
		if doModelAveraging:
			cumulative = cumulative + theta
		else:
			cumulative = theta
		# All calculations done -- stop the stopwatch
		toc = t.perf_counter()
		totTime = totTime + (toc - tic)
		# If model averaging is being done, need to calculate current objective value a bit differently
		if doModelAveraging:
			objValSeries.append( objFunc( cumulative/(it+2) ) )
		else:
			objValSeries.append( objFunc( cumulative ) )
		timeSeries.append( totTime )
	
	# Clean up the final model
	if doModelAveraging:
		final = cumulative / (horizon + 1)
	else:
		final = cumulative
		
	# Return the final model and the objective values obtained at various time steps
	return (final, objValSeries, timeSeries)

# For cyclic mode, the state is a tuple of the current coordinate and the number of dimensions
def getCyclicCoord( state ):
	curr = state[0]
	d = state[1]
	if curr >= d - 1 or curr < 0:
		curr = 0
	else:
		curr += 1
	state = (curr, d)
	return (curr, state)

# For random mode, the state is the number of dimensions
def getRandCoord( state ):
	d = state
	curr = random.randint( 0, d - 1 )
	state = d
	return (curr, state)

# For randperm mode, the state is a tuple of the random permutation and the current index within that permutation
def getRandpermCoord( state ):
	idx = state[0]
	perm = state[1]
	d = len( perm )
	if idx >= d - 1 or idx < 0:
		idx = 0
		perm = np.random.permutation( d )
	else:
		idx += 1
	state = (idx, perm)
	curr = perm[idx]
	return (curr, state)

# Get functions that offer various coordinate selection schemes
def coordinateGenerator( mode, d ):
	if mode == "cyclic":
		return (getCyclicCoord, (0,d))
	elif mode == "random":
		return (getRandCoord, d)
	elif mode == "randperm":
		return (getRandpermCoord, (0,np.random.permutation( d )))

# Given a coordinate update oracle and a coordinate selection oracle, implement coordinate methods
# The method returns the final model and the objective value acheived by the intermediate models
# This can be used to implement coordinate descent or ascent as well as coordinate minimization or
# maximization methods simply by modifying the coordinate update oracle
def doSDCM( coordUpdateFunc, getCoordFunc, objFunc, init, horizon = 10 ):
	objValSeries = []
	timeSeries = []
	totTime = 0
	selector = getCoordFunc[0] # Get hold of the function that will give me the next coordinate
	state = getCoordFunc[1] # The function needs an internal state variable - store the initial state
	
	# Initialize model as well as some bookkeeping variables
	alpha = init
	
	for it in range( horizon ):
		# Start a stopwatch to calculate how much time we are spending
		tic = t.perf_counter()
		
		# Get the next coordinate to update and update that coordinate
		(i, state) = selector( state )
		alpha[i] = coordUpdateFunc( alpha, i, it )

		toc = t.perf_counter()
		totTime = totTime + (toc - tic)
		
		objValSeries.append( objFunc( alpha ) )
		timeSeries.append( totTime )
		
	return (alpha, objValSeries, timeSeries)
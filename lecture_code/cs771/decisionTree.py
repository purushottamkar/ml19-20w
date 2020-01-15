'''
	Package: cs771
	Module: decisionTree
	Author: Puru
	Institution: CSE, IIT Kanpur
	License: GNU GPL v3.0
	
	Give skeletal support for implementing decision trees as well as plotting them in toy 2D settings
'''

import numpy as np
from matplotlib import pyplot as plt
import warnings

class Node:
	# A node stores its own depth (root = depth 0), its decision stump, its parent and child information
	# A node also stores the features along which its ancestors were split
	# Leaf nodes also store a constant label that is assigned to every data point that reaches that leaf
	def __init__( self, depth = 0, stump = (0,0), parent = None ):
		self.depth = depth
		self.stump = stump
		self.parent = parent
		self.left = None
		self.right = None
		self.isLeaf = True
		self.label = 0
		self.ancestorSplitFeats = np.empty( [0,], dtype = int )
		
	# For feature stumps, return the feature and the threshold used
	# This method may give unexpected results if used on non-feature stumps e.g. CSVM stumps
	def extractStumpModel( self ):
		# First figure out the threshold used in this stump
		threshold = -self.stump( np.array([0,0])[np.newaxis] )
		# Next, figure out the feature used in this stump
		if self.stump( np.array([1,0])[np.newaxis] ) == -threshold:
			feature = 1
		else:
			feature = 0
		return (feature, threshold)
		
	def predict( self, data ):
		# If I am a leaf I can predict rightaway
		# May change this constant leaf action to something more interesting and powerful
		if self.isLeaf:
			return self.label
		else:
			discriminant = self.stump( data )
			
			# Else I have to ask one of my children to do the job
			if discriminant > 0:
				return self.right.predict( data )
			else:
				return self.left.predict( data )
			
	# A stump generator should take data (X, y) and the list of features along which ancestors were split
	# The stump generator must return a stump which is a two tuple of (feature, threshold)
	def train( self, X, y, stumpGenerator, maxLeafSize, maxDepth ):
		# If too few data points are present, or else if this node is too deep in the tree, make it a leaf
		if y.size < maxLeafSize or self.depth >= maxDepth:
			self.isLeaf = True
			self.label = np.mean( y )
		# Throw a warning in case we try to split a shallow but well populated node that is pure as well
		elif np.unique(y).size < 2:
			warnings.warn( "Warning: attempt to split a pure node made. Node converted to leaf instead", UserWarning )
			self.isLeaf = True
			self.label = np.mean( y )
		else:
			# This node will be split and hence it is not a leaf
			self.isLeaf = False
			
			# Get the best possible decision stump
			self.stump = stumpGenerator( X, y, self.ancestorSplitFeats )
			(feature, threshold) = self.extractStumpModel()
			
			self.left = Node( depth = self.depth + 1, parent = self )
			self.left.ancestorSplitFeats = np.append( self.ancestorSplitFeats, feature )
			self.right = Node( depth = self.depth + 1, parent = self )
			self.right.ancestorSplitFeats = np.append( self.ancestorSplitFeats, feature )
			
			# Find which points go to my left child and which go to my right child
			discriminant = self.stump( X )
			# Train my two children recursively
			self.left.train( X[discriminant <= 0, :], y[discriminant <= 0], stumpGenerator, maxLeafSize, maxDepth )
			self.right.train( X[discriminant > 0, :], y[discriminant > 0], stumpGenerator, maxLeafSize, maxDepth )
	
	# This method is guaranteed to work properly only when all stumps are feature stumps
	# i.e. they use a single feature and a threshold to split a node. Using this method
	# on trees which use arbitrary linear classifiers as stumps may give unpredictable results
	def drawNodeSplits( self, fig, xlim, ylim ):
		if not self.isLeaf:
			plt.figure( fig.number )
			(feature, threshold) = self.extractStumpModel()
			
			# Is this a vertical split or a horizontal one?
			if feature == 0:
				plt.plot( [threshold, threshold], ylim, color = 'c', linestyle = '--' )
				self.left.drawNodeSplits( fig, [xlim[0], threshold], ylim )
				self.right.drawNodeSplits( fig, [threshold, xlim[1]], ylim )
			elif feature == 1:
				plt.plot( xlim, [threshold, threshold], color = 'c', linestyle = '--' )
				self.left.drawNodeSplits( fig, xlim, [ylim[0], threshold] )
				self.right.drawNodeSplits( fig, xlim, [threshold, ylim[1]] )

class Tree:
	def __init__( self, maxLeafSize = 10, maxDepth = 5 ):
		self.maxLeafSize = maxLeafSize
		self.maxDepth = maxDepth
		self.root = Node()
		
	def predict( self, xt, yt ):
		return self.root.predict( np.array( [xt, yt] )[np.newaxis] )
	
	def train( self, X, y, stumpGenerator ):
		self.root.train( X, y, stumpGenerator, self.maxLeafSize, self.maxDepth )

	def drawTreeSplits( self, fig, xlim, ylim ):
		self.root.drawNodeSplits( fig, xlim, ylim )
import numpy as np

def batch_iterator(X,y = None,batch_size = 64):
	"""Generator function that divides data into batches for mini-batch SGD"""
	points = X.shape()[0]
	for i in np.arange(0,points,batch_size):
		begin,end = i,min(i+batch_size,points)
		if y is not None:
			yield X[begin:end],y[begin:end]
		else:
			yield X[begin:end]
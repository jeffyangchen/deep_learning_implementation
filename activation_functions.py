import math

#NonLinear activation functions

class Activation(object):
	"""Abstract Class for activation function"""
	def function(self):
		"""
		activation function
		:return:
		"""
		raise NotImplementedError()

	def derivative(selfself):
		"""
		activation derivative
		:return:
		"""
		raise NotImplementedError()

def sigmoid(x):
	"""Sigmoid Activation"""
	return 1 / (1 + math.pow(math.e,-x))

def d_sigmoid(x):
	"""Derivative of Sigmoid"""
	return sigmoid(x) * sigmoid(1-x)

def RLU(x):
	"""Rectified Linear Unit (RLU) Activation"""
	if x <= 0:
		return 0
	else:
		return x

def d_RLU(x):
	"""Derivative of RLU"""
	if x <= 0:
		return 0
	else:
		return 1

def leaky_RLU(x,alpha = 0.01):
	"""Parameterized RLU Activation"""
	if x <= 0:
		alpha * x
	else:
		return x

def d_leaky_RLU(x,alpha = 0.01):
	"""Derivitave of Parameterized RLU"""
	if x <= 0:
		return alpha
	else:
		return 1

def softmax(x):
	"""Softmax activation"""
	return math.log(1 + math.pow(math.e,x))

def d_softmax(x):
	"""Derivative of softmax activaiton """
	return sigmoid(x)
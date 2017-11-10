import math
import numpy as np

#NonLinear activation functions

class Activation(object):
	"""Abstract Class for activation function"""
	def derivative(self):
		"""
		activation derivative
		:return:
		"""
		raise NotImplementedError()

	def name(self):
		return self.__class__.__name__

class Identity(object):
	def __call__(self,z):
		return z

	def derivative(self,z):
		return np.ones(z.shape)

class sigmoid(Activation):

	def __call__(self,z):
		return float(1) / 1 + np.exp(-z)

	def derivative(self,z):
		return (float(1) / 1 + np.exp(-z)) * (float(1) / 1 + np.exp(1 + z))

class RLU(Activation):

	def __call__(self, z):
		return np.where(z <= 0,0,z)

	def derivative(self, z):
		return np.where(z <= 0,0,1)

class Leaky_RLU(Activation):

	def __init__(self,alpha = 0.1):
		self.alpha = alpha

	def __call__(self, z):
		return np.where(x<=0,self.alpha*z,z)

	def derivative(self, z):
		return np.where(x<=0,self.alpha,1)

class softplus(Activation):

	def init(self):
		pass

	def __call__(self,z):
		return np.log(1+np.exp(z))

	def derivative(self,z):
		return sigmoid(z)

# def sigmoid(x):
# 	"""Sigmoid Activation"""
# 	return 1 / (1 + math.pow(math.e,-x))
#
# def d_sigmoid(x):
# 	"""Derivative of Sigmoid"""
# 	return sigmoid(x) * sigmoid(1-x)
#
# def RLU(x):
# 	"""Rectified Linear Unit (RLU) Activation"""
# 	if x <= 0:
# 		return 0
# 	else:
# 		return x
#
# def d_RLU(x):
# 	"""Derivative of RLU"""
# 	if x <= 0:
# 		return 0
# 	else:
# 		return 1
#
# def leaky_RLU(x,alpha = 0.01):
# 	"""Parameterized RLU Activation"""
# 	if x <= 0:
# 		alpha * x
# 	else:
# 		return x
#
# def d_leaky_RLU(x,alpha = 0.01):
# 	"""Derivitave of Parameterized RLU"""
# 	if x <= 0:
# 		return alpha
# 	else:
# 		return 1
#
# def softplus(x):
# 	"""Softmax activation"""
# 	return math.log(1 + math.pow(math.e,x))
#
# def d_softplus(x):
# 	"""Derivative of softmax activaiton """
# 	return sigmoid(x)
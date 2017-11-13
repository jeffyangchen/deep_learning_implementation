import math
import numpy as np

#NonLinear activation functions

class Activation_Function(object):
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

class Sigmoid(Activation_Function):

	def __call__(self,z):
		return float(1) / (1 + np.exp(-z))

	def derivative(self,z):
		return self.__call__(z) * self.__call__(-z)

class RLU(Activation_Function):

	def __call__(self, z):
		return np.where(z <= 0,0,z)

	def derivative(self, z):
		return np.where(z <= 0,0,1)

class Leaky_RLU(Activation_Function):

	def __init__(self,alpha = 0.1):
		self.alpha = alpha

	def __call__(self, z):
		return np.where(z<=0,self.alpha*z,z)

	def derivative(self, z):
		return np.where(x<=0,self.alpha,1)

class Softplus(Activation_Function):

	def init(self):
		pass

	def __call__(self,z):
		return np.log(1+np.exp(z))

	def derivative(self,z):
		return float(1) / (1 + np.exp(-z))

class Softmax(Activation_Function):
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def derivative(self, x):
        p = self.__call__(x)
        return p * (1 - p)

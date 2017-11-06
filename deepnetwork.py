import numpy as np
from deepnetwork_layer import *
from learning_optimizers import *
from loss_functions import *

class neural_network(object):
	"""
	neural network is determined by its shape i.e. number of hidden layers, number of activations in hidden layer
	"""
	def __init__(self,optimizer,loss_function):
		self.layers = []
		self.optimizer = optimizer
		self.loss_function = loss_function

	def add_layer(self,layer):
		if self.layers:
			layer.set_input_dim(shape = self.layers[-1].output_shape())
		layer.initialize()
		self.layers.append(layer)
import numpy as np
from deepnetwork_layer import *
from learning_optimizers import *
from loss_functions import *

class deep_network(object):
	"""
	network archeticture is determined by structure of individual layers
	"""
	def __init__(self,optimizer,loss_function):
		self.layers = []
		self.optimizer = optimizer
		self.loss_function = loss_function

	def add_layer(self,layer,n_neurons):
		if self.layers:
			layer.set_input_dim(shape = self.layers[-1].output_shape())
		if hasattr(layer,'initialize_weights'):
			layer.initialize_weights(n_neurons = n_neurons,optimizer = self.optimizer)
		self.layers.append(layer)

	def intialize_network_layers(self,shape,layer_type,optimizer):
		"""More convenient way to define network shape by adding multiple layers"""
		for row in shape:
			self.add_layer(layer_type,optimizer,row)

	def forward_pass(self,X):
		previous_layer_output = X
		for layer in self.layers:
			previous_layer_output = layer.forward_pass(previous_layer_output,training = True)
		return previous_layer_output

	def backward_pass(self,loss_grad):
		previous_error = loss_grad
		for layer in reversed(self.layers):
			previous_error = layer.backward_pass(previous_error)

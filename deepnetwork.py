import numpy as np
from deepnetwork_layer import *
from learning_optimizers import *
from loss_functions import *
from util import *
from terminaltables import AsciiTable

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

	def forward_propogation(self,X):
		previous_layer_output = X
		for layer in self.layers:
			previous_layer_output = layer.forward_pass(previous_layer_output,training = True)
		return previous_layer_output

	def backward_propogation(self,loss_grad):
		previous_error = loss_grad
		for layer in reversed(self.layers):
			previous_error = layer.backward_pass(previous_error)

	def batch_train(self,X,y):
		y_pred = self.forward_pass(X)
		loss = self.loss_function.loss(y,y_pred)
		error = self.loss_function.gradient(y,y_pred)
		self.backward_pass(error)
		return loss

	def fit(self,X,y,n_epochs,batch_size):
		"""Trains using batch SGD for a number of epochs"""
		for _ in range(n_epochs):
			batch_error = []
			for X_batch,y_batch in batch_iterator:
				loss = self.batch_train(X_batch,y_batch)
				batch_error.append(loss)

	def summary(self):
		print AsciiTable([['Model Summary']])
		print 'Data Input Shape %s' % str(self.layers[0].input_shape)

		table_data = [["Layer Type","Number of Hidden Units","Number of Parameters","Output Shape"]]
		total_params = 0
		for layer in self.layers:
			layer_name = layer.layer_name()
			params =layer.parameters()
			out_shape = layer.outshape()
			hidden_units = layer.n_neurons
			table_data.append([layer_name,str(hidden_units),str(params),str(out_shape)])
			total_params += params

		print AsciiTable(table_data).table
		print 'Number of Total Parameters: %d \n' % total_params

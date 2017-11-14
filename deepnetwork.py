import numpy as np
from deepnetwork_layer import *
from activation_functions import *
from learning_optimizers import *
from loss_functions import *
from util import *
from terminaltables import AsciiTable

class neural_net(object):
	"""
	network archeticture is determined by structure of individual layers
	"""
	def __init__(self,input_features = None,optimizer = None,loss_function = None,validation_set = None):
		self.layers = []
		self.input_features = input_features
		self.optimizer = optimizer
		self.loss_function = loss_function()
		self.errors = {'training':[],'validation':[]}
		self.accuracy = {'training':[],'validation':[]}
		#self.progressbar = progressbar.ProgressBar(widgets = bar_widgets)
		self.validation_set = validation_set
		if self.validation_set is not None:
			self.X_val = self.validation_set[0]
			self.y_val = self.validation_set[1]

	def add_layer(self,layer):
		if self.layers:
			layer.set_input_shape(input_shape = self.layers[-1].output_shape())
		else:
			layer.set_input_shape(input_shape = (self.input_features,))

		if hasattr(layer,'initialize_weights'):
			layer.initialize_weights(optimizer = self.optimizer)

		self.layers.append(layer)

	def intialize_network_layers(self,shape,layer_type):
		"""More convenient way to define network shape by adding multiple layers"""
		for row in shape:
			self.add_layer(layer_type(n_neurons = row))

	def forward_propogation(self,X):
		previous_layer_output = X
		for layer in self.layers:
			previous_layer_output = layer.forward_pass(previous_layer_output)
		return previous_layer_output

	def back_propogation(self,loss_grad):
		previous_error = loss_grad
		for layer in reversed(self.layers):
			previous_error = layer.backward_pass(previous_error)

	def batch_train(self,X,y):
		y_pred = self.forward_propogation(X)
		loss = np.mean(self.loss_function.loss(y,y_pred))
		error = self.loss_function.gradient(y,y_pred)
		accuracy = self.loss_function.accuracy(y,y_pred)
		#print 'error shape',error.shape
		self.back_propogation(error)
		return loss,accuracy

	def batch_test(self,X,y):
		y_pred = self.forward_propogation(X)
		loss = np.mean(self.loss_function.loss(y, y_pred))
		accuracy = self.loss_function.accuracy(y, y_pred)
		return loss, accuracy

	def fit(self,X,y,n_epochs,validation_test = False):
		"""Trains using batch SGD for a number of epochs"""
		#for _ in range(n_epochs):
		for _ in range(n_epochs):
			batch_error = []
			for X_batch,y_batch in batch_iterator(X,y):
				loss,accuracy = self.batch_train(X_batch,y_batch)
				batch_error.append(loss)
			if validation_test == True:
				if self.validation_set == None:
					print 'Validation Set Needed! (Set Validation_test to False on fit)'
				else:
					validation_error,validation_accuracy = self.batch_test(self.X_val,self.y_val)
					_,training_accuracy = self.batch_test(X,y)
					self.errors['validation'].append(validation_error)
					self.accuracy['validation'].append(validation_accuracy)
					self.accuracy['training'].append(training_accuracy)
			self.errors['training'].append(np.mean(batch_error))

	def summary(self):
		print 'Data Input Shape %s' % str(self.layers[0].input_shape)
		print AsciiTable([['Model Summary']]).table
		table_data = [["Layer Type","Number of Parameters","Output Shape"]]
		total_params = 0
		for layer in self.layers:
			layer_name = layer.layer_name()
			params = layer.parameters()
			out_shape = layer.output_shape()
			#hidden_units = layer.n_neurons
			table_data.append([layer_name,str(params),str(out_shape)])
			total_params += params

		print AsciiTable(table_data).table
		print 'Number of Total Parameters: %d \n' % total_params



if __name__ == '__main__':
	net0 = neural_net(10, SGD, Cross_Entropy)

	net0.intialize_network_layers([5,3], Connected, RLU)

	net0.add_layer(Connected(n_neurons = 3, activation_function = sigmoid))
	net0.summary()
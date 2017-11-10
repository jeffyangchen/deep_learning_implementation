import numpy as np
import math
import copy
from learning_optimizers import SGD
from loss_functions import l2,cross_entropy

class Layer(object):
    """Abstract Class for Connected Layer"""
    def set_input_shape(self,input_shape):
        """
        Sets dimension of the inputs to this layer. Argument is a tuple (dim,)
        :param dim:
        :return:
        """
        self.input_shape = input_shape

    def layer_name(self):
        """
        Name of layer used in model summary
        :return:
        """
        return self.__class__.__name__

    def parameters(self):
        """ Returns number of trainable parameters"""
        return 0

    def forward_pass(self,X,training):
        """Forward pass through network"""
        raise NotImplementedError()

    def backward_pass(self,acc_grad):
        """Backward pass through network"""
        raise NotImplementedError()

    def output_shape(self):
        """Shape of output produced by forward pass"""
        raise NotImplementedError

class Feedforward(Layer):
    """
    A fully connected feedforward network layer
    Parameters:
    n_units: int; number of neurons in layer
    input_shape: tuple; Input shape of the layer.
    """

    def __init__(self,n_neurons = 0,input_shape = None,activation_function = None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        self.W = None
        self.w_0 = None
        self.activation_function = activation_function()
        self.trainable = True

    def initialize_weights(self,optimizer):
        #Initialize Weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_neurons))
        self.w0 = np.zeros((1, self.n_neurons))
        self.W_optimizer = copy.copy(optimizer)
        self.wo_optimizer = copy.copy(optimizer)

    def parameters(self):
        #Returns the number of parameters
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self,X):
        """
        Calculate the output of current layer and pass to next layer.
        Saves activation terms to be used in backpropogation
        :param X: The output of the previous layer
        :param A: The ACTIVATION of the previous layer (output before activation function is applied)
        :return:
        """
        self.layer_input = X
        self.activation_input = self.layer_input
        self.activation_input = A
        self.activation = np.dot(self.W,X) + self.w0
        return self.activation_function(self.activation),self.activation_input

    def backward_pass(self,previous_error):
        """
        Update weights in layer by backpropogation and pass error term to next layer
        :param acc_grad:
        :return:
        """
        W = self.W
        w0 = self.w0

        if self.trainable:
            error_term = np.dot(W,previous_error) * self.activation_function.derivative(self.activation)
            grad_w = np.mean(np.dot(self.activation_input,error_term),axis = 0)
            grad_w0 = np.mean(error_term,axis = 0)

        # Update the layer weights

        self.W = self.W_opt.update(self.W, grad_w)
        self.w0 = self.w0_opt.update(self.w0, grad_w0)

        return error_term,W,w0

    def output_shape(self):
        return(self.n_neurons,)

class Activation(Layer):
    """
    Layer that applies an activation function e.g. sigmoid or RLU
    """
    def __init__(self,activation_function):
        self.activation_function = activation_function()

    def forward_pass(self,X,training):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self,error):
        return error * self.activation_function.derivative(self.layer_input)

    def output_shape(self):
        return self.input_shape
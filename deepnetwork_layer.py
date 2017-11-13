import numpy as np
import math

class Layer(object):
    """Abstract Class for Layer"""
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

class Connected(Layer):
    """
    A fully connected network layer
    Parameters:
    n_units: int; number of neurons in layer
    input_shape: tuple; Input shape of the layer.
    """

    def __init__(self,n_neurons = 0,input_shape = None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        self.W = None
        self.w_0 = None
        #self.trainable = True

    def initialize_weights(self,optimizer):
        #Initialize Weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_neurons))
        self.w0 = np.zeros((1, self.n_neurons))
        self.W_optimizer = optimizer
        self.w0_optimizer = optimizer

    def parameters(self):
        #Returns the number of parameters
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self,X,training = True):
        """
        Calculate the output of current layer and pass to next layer.
        Saves activation terms to be used in backpropogation
        :param X: The output of the previous layer. Usually the output of the non-linear activation function.
        :return:
        """
        self.layer_input = X
        # print 'X Shape',X.shape
        # print 'W Shape',self.W.shape
        # print 'W0 Shape',self.w0.shape
        self.activation = np.dot(X,self.W) + self.w0
        return self.activation

    def backward_pass(self,previous_error):
        """
        Update weights in layer and pass error term to next layer
        :param previous_error:
        :return:
        """
        W = self.W
        w0 = self.w0
        grad_w = np.dot(self.layer_input.T,previous_error)
        grad_w0 = np.sum(previous_error,axis = 0,keepdims = True)
        #print 'W shape',W.shape
        #print 'Layer Input shape',self.layer_input.shape
        #print 'previus_error.shape',previous_error.shape
        #print 'grad_w shape',grad_w.shape
        #print 'grad_w0 shape',grad_w0.shape
        # Update the layer weights

        self.W = self.W_optimizer.update(self.W, grad_w)
        self.w0 = self.w0_optimizer.update(self.w0, grad_w0)

        return np.dot(previous_error,W.T)

    def output_shape(self):
        return(self.n_neurons,)

class Activation(Layer):
    """
    Layer that applies an activation function e.g. sigmoid or RLU
    """
    def __init__(self,activation_function = None):
        self.activation_function = activation_function()

    def layer_name(self):
        return self.activation_function.__class__.__name__

    def forward_pass(self,X,training = True):
        self.layer_input = X
        return self.activation_function(X)

    def backward_pass(self,error):
        return error * self.activation_function.derivative(self.layer_input)

    def output_shape(self):
        return self.input_shape
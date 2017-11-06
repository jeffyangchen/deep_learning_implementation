import numpy as np
import math
import copy
from learning_optimizers import SGD
from loss_functions import l2,cross_entropy

class Layer(object):
    """Abstract Class for Connected Layer"""
    def set_input_dim(self,input_dim):
        """
        Sets dimension of the inputs to this layer
        :param dim:
        :return:
        """
        self.input_dim = input_dim

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

    def output_shape(selfself):
        """Shape of output produced by forward pass"""
        raise NotImplementedError

class Feedforward(Layer):
    """
    A fully connected feedforward network layer
    Parameters:
    n_units: int; number of neurons in layer
    input_shape: tuple; Input shape of the layer.
    """

    def __init__(self,n_neurons,input_shape,activation_function,activation_function_derivative):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        self.W = None
        self.w_0 = None
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.trainable = True

    def initialize_weights(self,optimizer,n_neurons):
        #Initialize Weights
        self.n_neurons = n_neurons
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_neurons))
        self.w0 = np.zeros((1, self.n_neurons))
        self.W_optimizer = copy.copy(optimizer)
        self.wo_optimizer = copy.copy(optimizer)

    def forward_pass(self,X,A):
        """
        Calculate the output of current layer and pass to next layer.
        Saves activation terms to be used in backpropogation
        :param X: The output of the previous layer
        :param A: The ACTIVATION of the previous layer (output before activation function is applied)
        :return:
        """
        self.layer_input = X
        self.activation_input = A
        self.activation = np.dot(self.W,X) + self.w0
        return self.activation_function(self.activation)

    def backward_pass(self,previous_error):
        """
        Update weights in layer by backpropogation and pass error term to next layer
        :param acc_grad:
        :return:
        """
        W = self.W
        w0 = self.w0

        if self.trainable:
            error_term = np.dot(W,previous_error) * self.activation_function_derivative(self.activation)
            grad_w = np.dot(self.activation_input,error_term)
            grad_w0 = error_term

        # Update the layer weights

        self.W = self.W_opt.update(self.W, grad_w)
        self.w0 = self.w0_opt.update(self.w0, grad_w0)

        return error_term,W,w0

    def output_shape(self):
        return(self.n_neurons,)
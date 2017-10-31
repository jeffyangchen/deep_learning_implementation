import numpy as np
import math
from learning_optimizers import SGD
from loss_functions import l2_loss

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

class Connected(Layer):
    """
    A fully connected neural network layer
    Parameters:
    n_units: int; number of neurons in layer
    input_shape: tuple; Input shape of the layer.
    """

    def __init__(self,n_neurons,input_shape):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        self.W = None
        self.w_0 = None
        self.trainable = True

    def initialize_weights(self):
        #Initialize Weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_neurons))
        self.w0 = np.zeros((1, self.n_neurons))

    def forward_pass(self,X):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self,acc_grad):
        W = self.W

        if self.trainable:
            grad_w = self.layer_input.T.dot(acc_grad)
            grad_w0 = np.sum(acc_grad,axis = 0, keepdims = True)
        # Update the layer weights

        self.W = self.W_opt.update(self.W, grad_w)
        self.w0 = self.w0_opt.update(self.w0, grad_w0)

        acc_grad = acc_grad.dot(W.T)
        return acc_grad

    def output_shape(self):
        return(self.n_neurons,)
import numpy as np

def forward_pass(weight_matrix, input_vector):
    weight_matrix = np.asarray(weight_matrix)
    input_vector = np.asarray(input_vector)
    return np.matmul(weight_matrix, input_vector)

class neural_network(object):
    """
    neural network is determined by its shape i.e. number of hidden layers, number of activations in hidden layer
    """
    def __init__(self,shape):
        self.shape = shape
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = [[0 for activations in xrange(layer)] for layer in shape]

    def zero_errors(self):
        self.errors = [[[0 for activations in xrange(layer)] for layer in shape]]


# 1 Hidden layer with 3 dimensional input, 2 activations in layer 2, 1 activation in last activation
shape = [3,2,1]
first_net = neural_network([1,2,1])
#first_net.initialize_weights()
print first_net.weights

np.random.randn(input_dim,output_dim)

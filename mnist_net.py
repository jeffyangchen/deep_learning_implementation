from deepnetwork import deep_network
from deepnetwork_layer import *
from mnist_loader import load_data_wrapper
from activation_functions import *
from learning_optimizers import *
from loss_functions import *
from util import *

training_data,validation_data,test_data = load_data_wrapper()

X_train,y_train = unwind_data(training_data)
print len(X_train),len(y_train)
mnist_net = deep_network(input_features =784 ,optimizer = SGD, loss_function = cross_entropy)

mnist_net.add_layer(Feedforward(n_neurons = 10,activation_function = RLU))
mnist_net.add_layer(Feedforward(n_neurons = 10,activation_function = sigmoid))
mnist_net.summary()
mnist_net.fit(X_train,y_train)


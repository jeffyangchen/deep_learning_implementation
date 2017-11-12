from deepnetwork import deep_net
from deepnetwork_layer import *
from mnist_loader import load_data_wrapper
from activation_functions import *
from learning_optimizers import *
from loss_functions import *
from util import *
from sklearn import datasets

# data = datasets.load_digits()
# X = data.data
# y = data.target
# X_test = X
# y_test = y
# print X.shape
# print y.shape
# #
# def batch_iterator(X,y = None,batch_size = 64):
# 	"""Generator function that divides data into batches for mini-batch SGD"""
# 	points = X.shape[0]
# 	for i in np.arange(0,points,batch_size):
# 		begin,end = i,min(i+batch_size,points)
# 		if y is not None:
# 			yield X[begin:end],y[begin:end]
# 		else:
# 			yield X[begin:end]


training_data,validation_data,test_data = load_data_wrapper()

X,y = unwind_data(training_data)
X_test,y_test = unwind_data(test_data)

print X[0].shape
print y[0].shape
n_features = 784
mnist_net = deep_net(input_features = n_features, optimizer = SGD, loss_function = Cross_Entropy)
mnist_net.add_layer(Feedforward(n_neurons = 10))
mnist_net.add_layer(Activation(activation_function = RLU))
mnist_net.add_layer(Feedforward(n_neurons = 10))
mnist_net.add_layer(Activation(activation_function = Sigmoid))
mnist_net.summary()
mnist_net.fit(X,y,n_epochs = 1,batch_size = 256)

loss,accuracy = mnist_net.batch_test(X_test,y_test)
print accuracy
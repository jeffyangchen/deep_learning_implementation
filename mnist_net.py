from deepnetwork import deep_net
from deepnetwork_layer import *
from mnist_loader import load_data_wrapper
from activation_functions import *
from learning_optimizers import *
from loss_functions import *
from util import *
from sklearn import datasets

def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

data = datasets.load_digits()
X = data.data
y = to_categorical(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)
print X.shape
print y.shape

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


# training_data,validation_data,test_data = load_data_wrapper()
#
# X,y = unwind_data(training_data)
# X_test,y_test = unwind_data(test_data)
#
# print X[0].shape
# print y[0].shape

n_features = 64
mnist_net = deep_net(input_features = n_features, optimizer = SGD(learning_rate = 0.01,batch_size = 256), loss_function = Cross_Entropy)
mnist_net.add_layer(Feedforward(n_neurons = 100))
mnist_net.add_layer(Activation(activation_function = Sigmoid))
mnist_net.add_layer(Feedforward(n_neurons = 10))
mnist_net.add_layer(Activation(activation_function = Softmax))
mnist_net.summary()
mnist_net.fit(X,y,n_epochs = 50)

loss,accuracy = mnist_net.batch_test(X_test,y_test)
print accuracy
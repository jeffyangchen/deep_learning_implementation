from deepnetwork import neural_net
from deepnetwork_layer import *
from mnist_loader import load_data_wrapper
from activation_functions import *
from learning_optimizers import *
from loss_functions import *
from util import *
from sklearn import datasets

# data = datasets.load_digits()
# X = data.data
# y = to_categorical(data.target)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)
# print X.shape
# print y.shape

training_data,validation_data,test_data = load_data_wrapper()

X,y = unwind_data_wrapper(training_data)
X_test,y_test = unwind_data_wrapper(test_data)
X_val,y_val = unwind_data_wrapper(validation_data)

n_features = 784
mnist_net = neural_net(input_features = n_features, optimizer = SGD(learning_rate = 0.01, batch_size = 64), loss_function = Cross_Entropy,validation_set = (X_val,y_val))
mnist_net.add_layer(Connected(n_neurons = 100))
mnist_net.add_layer(Activation(activation_function = RLU))
mnist_net.add_layer(Connected(n_neurons = 10))
mnist_net.add_layer(Activation(activation_function = Softmax))
mnist_net.summary()
mnist_net.fit(X,y,n_epochs = 50,validation_test = True)

training_errors,training_accuracy = mnist_net.errors['training'],mnist_net.accuracy['training']
validation_errors,validation_accuracy = mnist_net.errors['validation'],mnist_net.accuracy['validation']

labels = ['Training Error','Validation Error']
plotter(range(len(training_errors)),[training_errors,validation_errors],labels = labels,title = 'Error Plot',filename = 'Error Plot.html')

labels = ['Training Accuracy','Validation Accuracy']
plotter(range(len(training_accuracy)),[training_accuracy,validation_accuracy],labels = labels,title = 'Accuracy Plot',filename = 'Accuracy Plot.html')

loss,accuracy = mnist_net.batch_test(X_test,y_test)
print accuracy
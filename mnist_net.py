import deepnetwork
from mnist_loader import load_data_wrapper



training_data,validation_data,test_data = load_data_wrapper()

print len(training_data[0][0])
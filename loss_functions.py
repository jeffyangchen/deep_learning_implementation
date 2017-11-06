import numpy as np

class Loss(object):
    def loss(self,y_true,y_pred):
        return NotImplementedError()

    def gradient(self,y,y_pred):
        return NotImplementedError()

    def acc(selfself,y,y_pred):
        return 0

class l2(Loss):
    def __init(self): pass

    def loss(self,y_true,y_pred):
        return 0.5 * np.power((y_pred - y_true),2)

    def gradient(self,y_true,y_pred):
        return y_pred-y_true

class cross_entropy(Loss):
    def __init(self): pass

    def loss(self,y_true,y_pred):
        return y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)

    def gradient(self,y_true,y_pred):
        return (y_true / y_pred) * (1-y_true) / (1-y_pred)
import numpy as np

class Loss(object):
    def loss(self,y_true,y_pred):
        return NotImplementedError()

    def gradient(self,y,y_pred):
        return NotImplementedError()

    def acc(selfself,y,y_pred):
        return 0


class l2_loss(Loss):
    def __init(self): pass

    def loss(self,y_true,y_pred):
        return 0.5 * np.power((y_true-y_pred),2)

    def gradient(self,y_true,y_pred):
        return y_true - y_pred
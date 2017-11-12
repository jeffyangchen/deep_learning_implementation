import numpy as np

class Loss(object):
    def loss(self,y_true,y_pred):
        return NotImplementedError()

    def gradient(self,y,y_pred):
        return NotImplementedError()

    def acc(selfself,y,y_pred):
        return 0

class L2(Loss):
    def __init(self): pass

    def loss(self,y_true,y_pred):
        return 0.5 * np.power((y_pred - y_true),2)

    def gradient(self,y_true,y_pred):
        return y_pred-y_true

class Cross_Entropy(Loss):
    def __init(self): pass

    def loss(self,y_true,y_pred):
        # try:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return  - y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)
        #except:
            # print 'y_true.shape',y_true.shape
            # print 'y_pred.shape',y_pred.shape

    def gradient(self,y_true,y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / y_pred) +  (1-y_true) / (1-y_pred)

    def accuracy(self,y_true,y_pred):
        total = float(np.shape(y_true)[0])
        # print 'y_pred shape',y_pred[0].shape
        # print y_pred[0]
        # print 'y_true shape',y_true[0].shape
        # print y_true[0]
        y_true = np.argmax(y_true,axis = 1)
        y_pred = np.argmax(y_pred,axis = 1)
        # print y_true[0]
        # print y_pred[0]
        return np.sum(y_true == y_pred,axis = 0) / total


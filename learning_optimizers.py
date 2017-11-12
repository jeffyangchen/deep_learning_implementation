import numpy as np

class SGD(object):
    """
    Stochastic Gradient Descent (SGD)
    """

    def __init__(self,learning_rate = 0.01,momentum = 0,batch_size = 1):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = None
        self.batch_size = batch_size

    def update(self,w,grad_w):
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
        self.w_updt = self.momentum * self.w_updt + (1-self.momentum) * grad_w
        return w - self.learning_rate * (1/self.batch_size) * self.w_updt
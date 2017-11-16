import numpy as np

# Learning Optimizers Reference: http://cs231n.github.io/neural-networks-3/#sgd

class SGD(object):
    """
    Stochastic Gradient Descent (SGD)
    """

    def __init__(self,learning_rate = 0.01,momentum = 0,batch_size = 1):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_updt = None
        self.batch_size = float(batch_size)

    def update(self,w,grad_w):
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
        self.w_updt = self.momentum * self.w_updt + (1-self.momentum) * grad_w
        return w - self.learning_rate * (1/self.batch_size) * self.w_updt

class AdaGrad(object):
    """
    Adapative Gradient Descent by Duchi et al.
    """
    def __init__(self,learning_rate = 0.01,batch_size = 1):
        self.learning_rate = learning_rate
        self.cache = None
        self.batch_size = float(batch_size)

    def update(self,w,grad_w):
        epsilon = 1e-5
        if self.cache is None:
            self.cache = np.zeros(np.shape(w))

        self.cache += np.square(grad_w)
        self.w_updt = grad_w/ (np.sqrt(self.cache) + epsilon)

        return w - self.learning_rate * (1/self.batch_size) * self.w_updt

class RMSProp(object):
    """
    Root Mean Square Propogation
    """

    def __init__(self,learning_rate = 0.01,decay_rate = 0.9,batch_size = 1):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.cache = None
        self.batch_size = float(batch_size)

    def update(self,w,grad_w):
        epsilon = 1e-5
        if self.cache is None:
            self.cache = np.zeros(np.shape(w))

        self.cache = self.decay_rate * self.cache + (1-self.decay_rate) * np.square(grad_w)
        self.w_updt = grad_w / (np.sqrt(self.cache) + epsilon)

        return w - self.learning_rate * (1 / self.batch_size) * self.w_updt

class Adam(object):
    """
    RMSprop with Momentum
    """

    def __init__(self, learning_rate=0.01,grad_decay_rate = 0.9 ,decay_rate=0.99, batch_size=1):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.grad_decay_rate = grad_decay_rate
        self.cache = None
        self.grad_cache = None
        self.batch_size = float(batch_size)

    def update(self, w, grad_w):
        epsilon = 1e-8
        if self.cache is None:
            self.cache = np.zeros(np.shape(w))
        if self.grad_cache is None:
           self.grad_cache = np.zeros(np.shape(w))

        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * np.square(grad_w)
        self.grad_cache = self.grad_decay_rate * self.grad_cache + (1-self.grad_decay_rate) * grad_w
        self.w_updt = self.grad_cache / (np.sqrt(self.cache) + epsilon)

        return w - self.learning_rate * (1 / self.batch_size) * self.w_updt
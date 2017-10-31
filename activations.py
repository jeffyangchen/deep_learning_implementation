import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + (math.e ** -x))

def derivative_sigmoid(x):
    return sigmoid(x) * sigmoid(1-x)


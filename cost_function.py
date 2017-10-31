import math

def sigmoid(x):
    return 1 / (1 + math.pow(math.e,-x))

def derivative_sigmoid(x):
    return sigmoid(x) * sigmoid(1-x)

def l2_cost(x,y):
    return (1/2) * (x-y) ** 2
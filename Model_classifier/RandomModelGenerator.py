from torch.utils.data import Dataset
import numpy as np
import math
from random import random

def class_option(option):
    return {
        0 : 'Gauss',
        1 : 'Gauss+Gauss',
        2 : 'Gauss+Gauss+Gauss',
        3 : 'Gauss+Gauss+Exp',
        4 : 'Gauss+Exp',
        5 : 'Exp',
        6 : '4G',
        7 : '5G',
        8 : '6G',
        9 : '7G',
    }[option]

def Gauss(x, A, u, sigma):
    return A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))
def Exp(x, B, b):
    return B * math.exp(-b * x)

def G(x, A, u, sigma ):
    for i in range(len(x)):
        x[i] = Gauss(x[i], A, u, sigma)
    return x

def E(x, B, b):
    for i in range(len(x)):
        x[i] = Exp(x[i], B, b)
    return x

def GG(x, A1, u1, sigma1, A2, u2, sigma2 ):
    for i in range(len(x)):
        x[i] = Gauss(x[i], A1, u1, sigma1) + Gauss(x[i], A2, u2, sigma2)
    return x

def GGG(x, A1, u1, sigma1, A2, u2, sigma2, A3, u3, sigma3  ):
    for i in range(len(x)):
        x[i] = Gauss(x[i], A1, u1, sigma1) + Gauss(x[i], A2, u2, sigma2) + Gauss(x[i], A3, u3, sigma3)
    return x

def GGE(x, A1, u1, sigma1, A2, u2, sigma2, B1, b1  ):
    #x_r = x.copy()
    for i in range(len(x)):
        x[i] = Gauss(x[i], A1, u1, sigma1) + Gauss(x[i], A2, u2, sigma2) + Exp(x[i], B1, b1)
    return x

def GE(x, A1, u1, sigma1, B1, b1  ):
    #x_r = x.copy()
    for i in range(len(x)):
        x[i] = Gauss(x[i], A1, u1, sigma1) + Exp(x[i], B1, b1)
    return x
def multiG(n, x):
    u, A, sigma = [],[], []
    for it in range(n):
        u.append( 1 + 8*random())
        A.append(  0.7 + 0.6*random())
        sigma.append(  0.2 + 0.5*random())
    for i in range(len(x)):
        sum = 0
        for it in range(n):
            sum = sum + Gauss(x[i], A[it], u[it], sigma[it])
        x[i] = sum
    return x

def data_generator(X, option_model):
    if option_model == 'Gauss':
        return G(X, 0.7 + 0.6*random(), 1 + 8*random(), 0.1 + 0.4*random())
    if option_model == 'Gauss+Gauss' :
        return GG(X, 0.7 + 0.6*random(), 1 + 8*random(), 0.1 + 0.4*random(),
                  0.7 + 0.6*random(), 1 + 8*random(), 0.1 + 0.4*random())
    if option_model == 'Gauss+Gauss+Gauss' :
        return GGG(X, 0.7 + 0.6*random(), 1 + 8*random(), 0.1 + 0.4*random(),
                   0.7 + 0.6*random(), 1 + 8*random(), 0.1 + 0.4*random(),
                   0.7 + 0.6*random(), 1 + 8*random(), 0.1 + 0.4*random())
    if option_model == 'Gauss+Gauss+Exp' :
        return GGE(X, 0.7 + 0.6*random(), 1 + 8*random(), 0.1 + 0.4*random(),
                   0.7 + 0.6*random(), 1 + 8*random(), 0.1 + 0.4*random(),
                   0.7 + 0.6*random(), 0.5 + random() )
    if option_model == 'Gauss+Exp':
        return GE(X, 0.7 + 0.6*random(), 1 + 8*random(), 0.1 + 0.4*random(),
                  0.7 + 0.6*random(), 0.5 + random())
    if option_model == 'Exp' :
        return E(X, 0.7 + 0.6*random(), 0.5 + random() )
    if option_model == '4G':
        return multiG(4, X)
    if option_model == '5G':
        return multiG(5, X)
    if option_model == '6G':
        return multiG(6, X)
    if option_model == '7G':
        return multiG(7, X)
def generate_data(X_data, option):

    # Gauss, class 0
    option_model = class_option(option)
    #print(G(X_data, 1, 2 + 6*random(), 0.2 + 0.5*random()))
    X_data = data_generator(X_data, option_model)
    #print(X_data)

    return X_data

def build_class_vector(option):
    option_model = class_option(option)
    class_vector = np.zeros(2)
    if option_model == 'Gauss':
        class_vector[0] = 1
        return class_vector
    if option_model == 'Gauss+Gauss' :
        class_vector[0] = 2
        return class_vector
    if option_model == 'Gauss+Gauss+Gauss' :
        class_vector[0] = 3
        return class_vector
    if option_model == 'Gauss+Gauss+Exp' :
        class_vector[0] = 2
        class_vector[1] = 1
        return class_vector
    if option_model == 'Gauss+Exp':
        class_vector[0] = 1
        class_vector[1] = 1
        return class_vector
    if option_model == 'Exp' :
        class_vector[1] = 1
        return class_vector
    if option_model == '4G' :
        class_vector[0] = 4
        return class_vector
    if option_model == '5G' :
        class_vector[0] = 5
        return class_vector
    if option_model == '6G' :
        class_vector[0] = 6
        return class_vector
    if option_model == '7G' :
        class_vector[0] = 7
        return class_vector

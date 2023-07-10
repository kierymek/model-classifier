from torch.utils.data import Dataset
import numpy as np
import math

def Gauss( x, A, u, sigma ):
    return  A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))

class ShapeCreator(Dataset):
    def __init__(self, X_data, function, params):

        self.X = X_data
        #self.X = [[gauss[]]]
        self.Y = params
        if function == "Gauss" :
            for i in range(len(self.X)):
                for j in range(len(self.X[0])):
                    self.X[i][j] = Gauss( self.X[i][j], 1, self.Y[i][0], self.Y[i][1])
        if function == "Gauss+Gauss":
            for i in range(len(self.X)):
                for j in range(len(self.X[0])):
                    self.X[i][j] = Gauss(self.X[i][j], 1, self.Y[i][0], self.Y[i][1])\
                                   + Gauss(self.X[i][j], 1, self.Y[i][2], self.Y[i][3])
        if function == "Gauss+Gauss+Exp":
            for i in range(len(self.X)):
                for j in range(len(self.X[0])):
                    self.X[i][j] = Gauss(self.X[i][j], 1, self.Y[i][0], self.Y[i][1])\
                                   + Gauss(self.X[i][j], 1, self.Y[i][2], self.Y[i][3]) \
                                   + math.exp(-0.2 * self.X[i][j])

                #print(self.Y[i])
           #self.X = [[ Gauss( x_i, 1, params[0], params[1]) for x_i in X] for X in self.X]
        #print(self.X)
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return { 'input' :  self.X[index],
                 'output' : self.Y[index]}

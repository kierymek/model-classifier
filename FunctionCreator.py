from torch.utils.data import Dataset
import numpy as np

class FunctionCreator(Dataset):
    def __init__(self, X_data, function, params):

        #self.X = [[x] for x in X_data]
        self.X =  X_data
        self.Y = np.zeros(len(X_data))
        if function == "linear" :
           self. Y = [[self.X[point][0] * params[0] + params[1]] for point in range(len(X_data))]
        self.Y = np.array(self.Y, dtype=np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return { 'input' :  self.X[index],
                 'output' : self.Y[index]}

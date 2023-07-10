import numpy as np
from random import random
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from FunctionCreator import FunctionCreator
from tqdm import tqdm

from matplotlib import pyplot as plt

class ENN():
    def __init__ (self, n_in, n_hidden1, n_hidden2, n_hidden3, n_out):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in,n_hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden1, n_hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden2, n_hidden3),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden3,n_out)
        )
        self.optimizer = Adam(self.model.parameters(), lr = 0.001)
        self.loss_vector = []

    def train_step(self, x, y, criterion):
        self.model.zero_grad()
        #print(x, y)
        x = x.float()
        y = y.float()

        output = self.model(x)
        loss = criterion(output, y)
        loss.backward()
        self.optimizer.step()

    def train(self, data, epochs, batch):
        data_train = DataLoader(dataset = data, batch_size = batch, shuffle = True)
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            if epoch%100 == 0:
                print("Training epoch: ", epoch)
            for dummy, batch in enumerate(data_train):
                x_train, y_train = batch['input'], batch['output']
                #print(y_train)
                self.train_step(x_train, y_train, criterion)
            loss = criterion(self.model(x_train.float()), y_train.float())
            self.loss_vector.append(loss.item())
def main():
    EvolutionalNN = ENN(1, 2, 2, 2, 1)
    #for param_tensor in EvolutionalNN.model.state_dict():
        #print(param_tensor, "\t", EvolutionalNN.model.state_dict()[param_tensor][0])

    X_data = [n_x + 3 * (random() - 0.5) for n_x in range(50)]
    #Y_tensor = torch.tensor(Y_tensor)
    Y_data = [[(2 * x_point  + 2 + 10 * (random() - 0.5) )] for x_point in X_data]
    X_data = [[x] for x in X_data]
    X_data = np.array(X_data, dtype = np.float32)
    Y_data = np.array(Y_data, dtype = np.float32)
    X_tensor = X_data
    X_tensor = torch.tensor(X_tensor)

    Dataset = FunctionCreator(X_data, "linear", [2, 2])
    EvolutionalNN.train(Dataset, 200, 5 )


    Y_NN = EvolutionalNN.model(X_tensor)
    Y_NN = Y_NN.detach().numpy()

    fig, ax = plt.subplots()
    ax.plot(X_data, Y_data, 'ko', label = "Data points")
    #print("Regression:")
    #print("A: ",A,", mean: ", u,", sigma: ", sigma)
    #X = np.linspace(0,50,501)
    #Y = [ gauss(x_point, A, u, sigma) for x_point in X]
    ax.plot(X_data, Y_NN, 'blue', label = "Evolutional NN")
    leg = ax.legend(loc = 'upper left', prop={'size':7})

    #A, u, sigma = round(A, 8), round(u, 8), round(sigma, 8)
    #a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    #ax.text(0.45, 1.06, "A: " + str(A) + ", mean: " + str(u) + ", sigma: " + str(sigma),
    #    horizontalalignment='center', verticalalignment='center',
    #    transform=ax.transAxes, color = 'blue')
    plt.show()

if __name__ == "__main__" :
    main()

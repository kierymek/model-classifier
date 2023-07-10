import numpy as np
import random
import torch
from matplotlib import pyplot as plt
import math

from ShapeCreator import ShapeCreator
from ENN_classifier import ENN_classifier
from RandomModelGenerator import generate_data

def Gauss( x, A, u, sigma ):
    return  A * np.exp(-np.power(x - u, 2) / (2 * np.power(sigma, 2)))


def main():
    epoch = 60
    train_samples = 1200
    test_samples = 200
    classes = 6
    EvolutionalNN = ENN_classifier(40, 40, 20, 10, classes)


    X_data = [[X/4 for X in range(40)] for it in range(classes*(train_samples + test_samples))]
    Y_data = [ 0 for X in X_data]
    X_data = np.array(X_data, dtype = np.float32)
    Y_data = np.array(Y_data, dtype = np.longlong)

    for c in range(classes):
        for i in range(train_samples + test_samples):
            X_data[i + c*(train_samples + test_samples)] = generate_data(X_data[i + c*(train_samples + test_samples)], c)
            Y_data[i + c*(train_samples + test_samples)] = c

    indices = np.arange(X_data.shape[0])
    np.random.shuffle(indices)
    #print(Y_data)
    X_data = X_data[indices]
    Y_data = Y_data[indices]

    X_train, Y_train = X_data[:(train_samples * classes)], Y_data[:(train_samples * classes)]
    X_test, Y_test = X_data[(train_samples * classes):], Y_data[(train_samples * classes):]

    Dataset = ShapeCreator(X_train, "dummy", Y_train)
    EvolutionalNN.train(Dataset, epoch, 20 )

    X_NN = torch.tensor(X_test)
    Y_NN = EvolutionalNN.model(X_NN)
    Y_NN = Y_NN.detach().numpy()
    #print(X_NN, Y_NN)
    #idx = np.zeros([classes * (train_samples), classes])
    idx_gge = []
    accuraccy = 0
    for i in range(len(Y_NN)):
        if Y_test[i] == 3:
            idx_gge.append(i)
        #print(Y_test[i], Y_NN[i, Y_test[i]])
        if Y_NN[i,Y_test[i]] == max(Y_NN[i]):
            accuraccy = accuraccy + 1
    accuraccy = accuraccy / (classes * test_samples)
    print("Reached accuraccy: ",accuraccy)

    #for it in range(len(Y_NN)):
    #    print(Y_NN[it], Y_test[it])
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(8, 8)
    n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 0], 100, alpha=0.5, range=[0, 1], label='Gauss')
    n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 1], 100, alpha=0.5, range=[0, 1], label='Gauss+Gauss')
    n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 2], 100, alpha=0.5, range = [0,1], label='Gauss+Gauss+Gauss')
    n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 3], 100, alpha=0.5, range = [0,1], label='Gauss+Gauss+Exp')
    n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 4], 100, alpha=0.5, range = [0,1], label='Gauss+Exp')
    n, bins, patches = ax[0, 0].hist(Y_NN[idx_gge, 5], 100, alpha=0.5, range=[0, 1],  label='Exp')
    ax[0, 0].set_xlabel('Class A (Gauss+Gauss+Exp)')
    ax[0, 0].set_ylabel('Number of counts')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].set_title('Class affiliation probability')

    ax[1, 0].plot(range(0,epoch), EvolutionalNN.loss_vector, 'k-', label="Loss function")
    ax[1, 0].set_xlabel('Epoch number')
    ax[1, 0].set_ylabel('Loss function')
    ax[1, 0].set_yscale('log')

    X_plot = [X / 40 for X in range(400)]
    X_plot = np.array(X_plot, dtype = np.float32)
    X_plot = generate_data(X_plot, 3)
    ax[0, 1].plot(np.linspace(0, 10, 400, endpoint=False), X_plot, 'r-', label="GGE(x)")
    ax[0, 1].set_xlabel('X argument')
    ax[0, 1].set_ylabel('GGE(x)')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].set_title('Exemplary training GGE(x) curve')
    #ax[0,1].plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[2], 'r-', label="Gauss no. 2")
    #ax[0,1].plot(np.linspace(0, 10, vis_len, endpoint=False), In_data[3], 'b-', label="Gauss no. 3")

    #ax[0,0].plot(Y_NN[1][0], Gauss(Y_NN[1][0], 1, Y_NN[1][0], Y_NN[1][1]), 'k*', label="Predicted mean")
    #ax[0,0].plot(Y_NN[2][0], Gauss(Y_NN[2][0], 1, Y_NN[2][0], Y_NN[2][1]), 'r*', label="Predicted mean")
    #ax[0,0].plot(Y_NN[3][0], Gauss(Y_NN[3][0], 1, Y_NN[3][0], Y_NN[3][1]), 'b*', label="Predicted mean")
    #ax[0,0].set_xlabel('Argument X')
    #ax[0,0].set_ylabel('Value Y')

    #leg = ax[0,0].legend(loc = 'upper left', prop={'size':7})

    #A, u, sigma = round(A, 8), round(u, 8), round(sigma, 8)
    #a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    #ax[0,0].text(0.45, 1.12, " Mean: " + str(round(Y_data[1][0],3)) + ", NN prediction: " + str(round(Y_NN[1][0],3)),
    #    horizontalalignment='center', verticalalignment='center',
    #    transform=ax[0,0].transAxes, color = 'k')
    #ax[0,0].text(0.45, 1.07, " Mean: " + str(round(Y_data[2][0], 3)) + ", NN prediction: " + str(round(Y_NN[2][0], 3)),
    #        horizontalalignment='center', verticalalignment='center',
    #        transform=ax[0,0].transAxes, color='r')
    #ax[0,0].text(0.45, 1.02, " Mean: " + str(round(Y_data[3][0], 3)) + ", NN prediction: " + str(round(Y_NN[3][0], 3)),
    #        horizontalalignment='center', verticalalignment='center',
    #        transform=ax[0,0].transAxes, color='b')

    #n, bins, patches = ax[0, 1].hist(Y_data[:,0] - Y_NN[:,0], 100, alpha=0.5, color = 'red', label='Mean')
    #ax[0, 1].set_xlabel('Real - NN prediction')
    #ax[0, 1].set_ylabel('Number of counts')

    #n, bins, patches = ax[1, 1].hist(Y_data[:,1] - Y_NN[:,1], 100, alpha=0.5, color = 'blue', label='St. dev.')
    #ax[1, 1].set_xlabel('Real - NN prediction')
    #ax[1, 1].set_ylabel('Number of counts')
    #axs[0, 1].axvline(x=np.mean(Mass_B_data), color='r', linestyle='dashed')
    #ax[0, 1].legend(loc='upper right')
    #ax[1, 1].legend(loc='upper right')

    #ax[1, 0].plot(range(0,epoch), EvolutionalNN.loss_vector, 'k-', label="Loss function")
    #ax[1, 0].set_xlabel('Epoch number')
    #ax[1, 0].set_ylabel('Loss function')
    #ax[1, 0].set_yscale('log')

    plt.show()

if __name__ == "__main__" :
    main()

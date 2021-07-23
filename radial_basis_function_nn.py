import torch
import torch.nn as nn
import radial_basis_functions as rbfs
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
import numpy as np
import matplotlib.pyplot as plt

'''
    Radial Basis Function Network
'''

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x,y)

class RBF(nn.Module):
    '''
        Parameters:
            in_features:  size of input samples
            out_features: size of output smaples
            basis_func:   radial basis function used to transform the scaled distances
    '''
    def __init__(self, in_features, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        # initializing centers from normal distributions with 0 mean and std 1
        nn.init.normal_(self.centers, 0,1)
        # initializing sigmas with 0 value
        nn.init.constant_(self.log_sigmas,0)
        

    def forward(self, x):
        '''Forward pass'''
        input = x
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5)/ torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)



# Class for dynamic Radial Basis Neural Network creation
class Network(nn.Module):

    def __init__(self, layer_widths, layer_centers, basis_func):
        super(Network, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths)-1):
            self.rbf_layers.append(RBF(layer_widths[i], layer_centers[i], basis_func))
            self.linear_layers.append(nn.Linear(layer_centers[i], layer_widths[i+1]))
    
    def forward(self, x):
        out = x
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out

    def fit(self, x, y, epochs, batch_size, lr, loss_func):
        self.train()
        obs = x.size(0)
        trainset = MyDataset(x,y)
        trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        epoch = 0
        
        # training loop
        while epoch < epochs:
            epoch += 1
            current_loss = 0
            batches = 0
            progress = 0

            #print('Epoch %s' % (epoch))

            # iterate over the data
            for x_batch, y_batch in trainloader:
                batches += 1

                # set gradients of all tensors to zero
                optimizer.zero_grad()

                # forward pass of data through net
                y_hat = self.forward(x_batch)

                # compute loss
                loss = loss_func(y_hat, y_batch)
                current_loss += (1/batches) * (loss.item() - current_loss)

                # backward pass
                loss.backward()

                # optimize parameters
                optimizer.step()

                # show stats
                #if(batches % 500 == 499):
                #    print('Loss after mini-batch %5d: %.3f'% (batches+1, current_loss/500))
                # save stats
                progress += y_batch.size(0)
                sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f      ' % \
                                 (epoch, progress, obs, current_loss))
                sys.stdout.flush()
                


if __name__ == '__main__':

    # Hyperparameters
    batch_size = 10
    learning_rate = 1e-4
    epochs = 5000
    layer_widths = [2, 1]
    layer_centres = [40]
    basis_func = rbfs.gaussian

    ##############
    # Toy Dataset
    #
    x1 = np.linspace(-1,1,101)
    # decision boundary
    x2 = 0.5*np.cos(np.pi*x1) + 0.5*np.cos(4*np.pi*(x1+1))

    samples = 200
    x = np.random.uniform(-1, 1, (samples, 2))
    for i in range(samples):
        if i < samples//2:
            x[i,1] = np.random.uniform(-1, 0.5*np.cos(np.pi*x[i,0]) + 0.5*np.cos(4*np.pi*(x[i,0]+1)))
        else:
            x[i,1] = np.random.uniform(0.5*np.cos(np.pi*x[i,0]) + 0.5*np.cos(4*np.pi*(x[i,0]+1)), 1)

    steps = 100
    x_span = np.linspace(-1, 1, steps)
    y_span = np.linspace(-1, 1, steps)
    xx, yy = np.meshgrid(x_span, y_span)
    values = np.append(xx.ravel().reshape(xx.ravel().shape[0], 1),
                    yy.ravel().reshape(yy.ravel().shape[0], 1),
                    axis=1)

    tx = torch.from_numpy(x).float()
    ty = torch.cat((torch.zeros(samples//2,1), torch.ones(samples//2,1)), dim=0)
    ##############

    # defining loss function
    #loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCEWithLogitsLoss()

    # Initializing Net
    rbnn = Network(layer_widths, layer_centres, basis_func)
    rbnn.fit(tx, ty, epochs, samples, learning_rate, loss_function)
    # end
    print('Training finished!')

    # plotting ground truth and estimation boundary
    with torch.no_grad():
        preds = (torch.sigmoid(rbnn(torch.from_numpy(values).float()))).data.numpy()
    ideal_0 = values[np.where(values[:,1] <= 0.5*np.cos(np.pi*values[:,0]) + 0.5*np.cos(4*np.pi*(values[:,0]+1)))[0]]
    ideal_1 = values[np.where(values[:,1] > 0.5*np.cos(np.pi*values[:,0]) + 0.5*np.cos(4*np.pi*(values[:,0]+1)))[0]]
    area_0 = values[np.where(preds[:, 0] <= 0.5)[0]]
    area_1 = values[np.where(preds[:, 0] > 0.5)[0]]

    fig, ax = plt.subplots(figsize=(16,8), nrows=1, ncols=2)
    ax[0].scatter(x[:samples//2,0], x[:samples//2,1], c='dodgerblue')
    ax[0].scatter(x[samples//2:,0], x[samples//2:,1], c='orange', marker='x')
    ax[0].scatter(ideal_0[:, 0], ideal_0[:, 1], alpha=0.1, c='dodgerblue')
    ax[0].scatter(ideal_1[:, 0], ideal_1[:, 1], alpha=0.1, c='yellow')
    ax[0].set_xlim([-1,1])
    ax[0].set_ylim([-1,1])
    ax[0].set_title('Ideal Decision Boundary')

    ax[1].scatter(x[:samples//2,0], x[:samples//2,1], c='dodgerblue')
    ax[1].scatter(x[samples//2:,0], x[samples//2:,1], c='orange', marker='x')
    ax[1].scatter(area_0[:, 0], area_0[:, 1], alpha=0.1, c='dodgerblue')
    ax[1].scatter(area_1[:, 0], area_1[:, 1], alpha=0.1, c='yellow')
    ax[1].set_xlim([-1,1])
    ax[1].set_ylim([-1,1])
    ax[1].set_title('RBF Decision Boundary')
    plt.show()

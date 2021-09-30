import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch.utils.data as data_utils

import numpy as np

import matplotlib.pyplot as plt

class SwapNoiseMasker(object):
    def __init__(self, probas):
        self.probas = torch.from_numpy(np.array(probas))

    def apply(self, X):
        should_swap = torch.bernoulli(self.probas.to(X.device) * torch.ones((X.shape)).to(X.device))
        corrupted_X = torch.where(should_swap == 1, X[torch.randperm(X.shape[0])], X)
        mask = (corrupted_X != X).float()
        return corrupted_X, mask

def print_scores(l_loss, l_roc_train,l_loss_intermediate, l_roc_intermediary, l_roc_test, l_loss_test):
    
    plt.rcParams['figure.figsize'] = [20 ,10]

    fig, ax = plt.subplots(nrows=3, ncols=2)


    ax[0,0].plot(l_loss)
    ax[0,1].plot(l_roc_train)
    ax[1,0].plot(l_loss_intermediate)
    ax[1,1].plot(l_roc_intermediary)
    ax[2,0].plot(l_loss_test)
    ax[2,1].plot(l_roc_test)

    ax[0,0].set_xlabel("Epochs")
    ax[0,0].set_ylabel("Loss")
    ax[0,0].set_title("Loss Train vs Epochs")

    ax[0,1].set_xlabel("Epochs")
    ax[0,1].set_ylabel("AUC")
    ax[0,1].set_title("AUC Train vs Epochs")
    
    ax[1,0].set_xlabel("Epochs")
    ax[1,0].set_ylabel("Loss")
    ax[1,0].set_title("Loss BCE vs Epochs")

    ax[1,1].set_xlabel("Epochs")
    ax[1,1].set_ylabel("AUC")
    ax[1,1].set_title("AUC BCE vs Epochs")

    ax[2,0].set_xlabel("Epochs")
    ax[2,0].set_ylabel("Loss")
    ax[2,0].set_title("Loss Test vs Epochs")

    ax[2,1].set_xlabel("Epochs")
    ax[2,1].set_ylabel("AUC")
    ax[2,1].set_title("AUC Test vs Epochs")
    
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)

    fig=plt.figure(figsize=(16,8), dpi= 100, facecolor='w', edgecolor='k')

    fig.tight_layout()
    

def pandas_to_tensor(df, emb_xs, emb_valid_xs):

    target = df.target

    train = emb_xs.join(target)
    train = train.reset_index(drop=True)
    train_data = torch.Tensor(train.drop(["target"], axis=1).values)
    train_target = torch.Tensor(train[["target"]].values)

    valid = emb_valid_xs.join(target)
    valid = valid.reset_index(drop=True)
    valid_data = torch.Tensor(valid.drop(["target"], axis=1).values)
    valid_target = torch.Tensor(valid[["target"]].values)

    train_temp = data_utils.TensorDataset(train_data, train_target)
    train_loader = data_utils.DataLoader(train_temp, batch_size=380)

    valid_temp = data_utils.TensorDataset(valid_data, valid_target)
    valid_loader = data_utils.DataLoader(valid_temp, batch_size=380)
    
    return train_loader, valid_loader
    
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        """
        To add dropout or other idk
        """
        self.fc1 = nn.Linear(in_features =input_size, out_features = 10)
        self.fc2 = nn.Linear(in_features = 20, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=output_size)
        self.bn1 = nn.BatchNorm1d(num_features=10)
        self.bn2 = nn.BatchNorm1d(num_features=10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        #x = self.relu(x)

        #x = F.glu(x)
        x = torch.sigmoid(self.fc3(x))
        #x = self.fc2(self.dropout(x))
        #x = self.bn2(x)
        #x = torch.sigmoid(self.fc3(self.dropout(x)))
        soft = nn.Softmax(dim=0)
        prob_labels = soft(x)
        return prob_labels, x

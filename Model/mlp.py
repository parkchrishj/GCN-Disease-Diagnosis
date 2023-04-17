import torch
import torch.nn as nn
import torch.nn.functional as F

'''
MLP with lienar output
The module has two modes of operation: linear and multi-layer. If num_layers is set to 1, the module functions as a linear model, 
using a single nn.Linear layer to produce the output. Otherwise, if num_layers is greater than 1, the module functions as an MLP with 
multiple hidden layers. The nn.Linear and nn.BatchNorm1d modules are used to define the linear layers and batch normalization layers 
in the MLP. The nn.ModuleList is used to store the linear layers and batch normalization layers as a list, allowing for easy access 
during forward pass. The forward method implements the forward pass of the MLP. If the module is operating in linear mode, it simply 
applies the linear layer to the input features x and returns the output. Otherwise, if the module is operating in multi-layer mode, 
it applies ReLU activation and batch normalization to the hidden layers, and then applies the final linear layer to produce the output.
'''

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

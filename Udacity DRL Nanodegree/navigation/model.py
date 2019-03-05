import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque



class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, hidden_layers, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers (list of int): number of nodes in each layer
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer_sizes = [state_size] + hidden_layers
        layer_dims = zip(layer_sizes[:-1], layer_sizes[1:])
        self.hidden_layers = nn.ModuleList([nn.Linear(h1, h2) 
                                           for h1, h2 in layer_dims])
        self.output = nn.Linear(layer_sizes[-1], action_size)
        
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        output = self.output(x)

        return output


 
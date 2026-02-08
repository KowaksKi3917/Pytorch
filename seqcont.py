# Neurl network with an input layer with 5 neurons, a hidden layers with 3 neurons (Relu) and one output neuron (Sigmoid)
import torch
import torch.nn as nn
from torchinfo import summary

class hiddennn(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.network=nn.Sequential(
        nn.Linear(num_features,3),
        nn.ReLU(),
        nn.Linear(3,1),
        nn.Sigmoid())
    def forward(self,features):
        out=self.network(features)

        return out
    
#Dataset
features=torch.rand(10,5)
model=hiddennn(features.shape[1])
print(model(features))
print(model.linear1.weight)
print(model.linear1.bias)
print(model.linear2.weight)
print(model.linear2 .bias)
summary(model,input_size=(10,5))
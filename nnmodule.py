import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.linear=nn.Linear(num_features,1)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,features):
         out=self.linear(features)
         out=self.sigmoid(out)

         return out 
    
# Creating a dataset
features=torch.rand(10,5)

# Creating a model
model=Model(features.shape[1])

# Forward pass
print(model(features))

print(model.linear.weight)
print(model.linear.bias)

# Visualization
from torchinfo import summary
summary(model, input_size=(10,5))

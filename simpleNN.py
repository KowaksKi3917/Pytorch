import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv")
# print(df.head())
# print(df.shape)
df=df.drop(columns=['id','Unnamed: 32'])
X_train, X_test, y_train, y_test=train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size=0.2)
scalar=StandardScaler( )
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)
encoder=LabelEncoder()
y_train=encoder.fit_transform(y_train)
y_test=encoder.transform(y_test)
# print(y_train)
# print(y_test)
X_train_tensor=torch.from_numpy(X_train)
X_test_tensor=torch.from_numpy(X_test)
y_train_tensor=torch.from_numpy(y_train)
y_test_tensor=torch.from_numpy(y_test)
#print(X_train_tensor.shape)
class MysimpleNN():
     def __init__(self,X): 
          self.weights=torch.rand(X.shape[1],1,dtype=torch.float64, requires_grad=True)
          self.bias=torch.zeros(1,dtype=torch.float64, requires_grad=True)
     def forward(self,X):
          z=torch.matmul(X,self.weights)+self.bias
          y_pred=torch.sigmoid(z)
          return(y_pred)
     def loss_function(self,y_pred,y):
          epsilon=1e-7
          y_pred=torch.clamp(y_pred, epsilon,1-epsilon)
          loss = -(y_train_tensor * torch.log(y_pred)+ (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
          return(loss)
           

learning_rate=0.9
epochs=25
#create a model
model= MysimpleNN(X_train)
# print(model.weights)
# loop
# forward pass
# loss calculation
# backward pass
# parameter update

for epoch in range(epochs):
     y_pred=model.forward(X_train_tensor)
     loss=model.loss_function(y_pred,y_train_tensor)
     loss.backward()
     with torch.no_grad():
          model.weights-=learning_rate*model.weights.grad
          model.bias-=learning_rate*model.bias.grad
     model.weights.grad.zero_()
     model.bias.grad.zero_()
     print(f'epoch: {epoch +1}, Loss: {loss.item()}')

# Evaluation
with torch.no_grad():
     y_pred=model.forward(X_test_tensor)
     y_pred=(y_pred>0.5).float()
     accuracy=(y_pred==y_test_tensor).float().mean()
     print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
     print(f"The accuracy is {accuracy}")
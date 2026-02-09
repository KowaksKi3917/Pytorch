import torch
import numpy as np
import pandas as pd
import torch.nn as nn
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
X_train_tensor=torch.from_numpy(X_train).float()
X_test_tensor=torch.from_numpy(X_test).float()
y_train_tensor=torch.from_numpy(y_train).float()
y_test_tensor=torch.from_numpy(y_test).float( )
# Custom dataset
from torch.utils.data import Dataset, DataLoader
class customdataset(Dataset):
     def __init__(self,features,labels):
          self.features=features
          self.labels=labels
     def __len__(self):
          return len(self.features)
     def __getitem__(self,index):
          return self.features[index] , self.labels[index]
# custom datasets    
train_dataset= customdataset(X_train_tensor,y_train_tensor)
test_dataset= customdataset(X_test_tensor,y_test_tensor)
# custom dataloaders
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True)
#print(X_train_tensor.shape)
class MysimpleNN(nn.Module):
     def __init__(self,num_features): 
        super().__init__()
        self.linear=nn.Linear(num_features,1)
        self.sigmoid=nn.Sigmoid()
     def forward(self,features):
          out=self.linear(features)
          out=self.sigmoid(out)
          return(out)
           
learning_rate=0.9 
epochs=25
# Loss function
loss_function=nn.BCELoss()
#create a model
model= MysimpleNN(X_train_tensor.shape[1])
# print(model.weights)
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate) # Optimizer
for epoch in range(epochs):
     for batch_features, batch_labels in train_loader:
          y_pred=model(batch_features)
          loss= loss_function(y_pred,batch_labels.view(-1,1  )) # loss calculation
          optimizer.zero_grad() 
          loss.backward()  # backward pass
          optimizer.step() # parameter update
          print(f'epoch: {epoch +1}, Loss: {loss.item()}')
     
# Evaluation
model.eval()
accuracy_list=[]
with torch.no_grad():
     for batch_features,batch_labels in test_loader:
        y_pred=model.forward(batch_features)
        y_pred=(y_pred>0.5).float()
        batch_accuracy=(y_pred.view(-1)==batch_labels).float().mean()
        accuracy_list.append(batch_accuracy)
     
overall_accuracy=sum(accuracy_list)/len(accuracy_list)
print(f"Accuracy : {overall_accuracy:.4f}")
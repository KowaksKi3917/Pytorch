import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import optuna
# Seeding
torch.manual_seed(42)
# Defining GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device :{device}")
df=pd.read_csv("/Users/parikshitbhardwaj/Downloads/archive/fashion-mnist_train.csv")
# print(df.head())
# # plotting some images
# fig, axes = plt.subplots(4,4, figsize=(10,10))
# fig.suptitle("First 16 imsges", fontsize=16)
# for i, ax in enumerate(axes.flat):
#      img = df.iloc[i, 1:]. values. reshape(28, 28) # Reshape to 28x28
#      ax. imshow(img)  # Display in grayscale ax.axis ('off') # Remove axis for a cleaner look ax. set_title(f"| shal. 'df. iloc [i, 0])") # Show the label
#      ax.axis('off')
#      ax.set_title(f"Label : {df.iloc[i,0]}")
# plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to fit the title pit. show()
# plt.show()
x=df.iloc[:,1:].values
y=df.iloc[:,0].values
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)
# Scaling
X_train=X_train/255.0 # The values in the dataset are spread between 0 to 255 (pixel values)
X_test=X_test/255.0
# Customdataset
class customdataset(Dataset):
     def __init__(self, features, labels):
          self.features=torch.tensor(features, dtype=torch.float32)
          self.labels=torch.tensor(labels, dtype=torch.long)
     def __len__(self):
          return len(self.features[X_train[0]])
     def __getitem__(self,index):
          return self.features[index], self.labels[index] 

train_dataset=customdataset(X_train,y_train)
test_dataset=customdataset(X_test,y_test)
class MyNN(nn.Module):
     def __init__(self, input_dim, output_dim, num_hidden_layers, num_neurons, dropout_rate):
          super().__init__()
          layers=[]
          for i in range(num_hidden_layers):
               layers.append(nn.Linear(input_dim,num_neurons))
               layers.append(nn.BatchNorm1d(num_neurons))
               layers.append(nn.ReLU())
               layers.append(nn.Dropout(dropout_rate))
               input_dim=num_neurons
          layers.append(nn.Linear(num_neurons,output_dim))
          self.model=nn.Sequential(*layers)
     def forward(self,x):
          return self.model(x)
    
#Objective Function
def objective(trial):
     #hyperparameter values from the search space
     num_hidden_layers=trial.suggest_int("num_hidden_layers",1,5)
     num_neurons=trial.suggest_int("num_neurons",8,128,step=8)
     epochs=trial.suggest_int("epochs",10,50,step=10)
     learning_rate=trial.suggest_float("learning_rate",1e-5,1e-1,log=True)
     dropout_rate=trial.suggest_float("dropout_rate",0.1,0.5,step=0.1)
     batch_size=trial.suggest_categorical("batch_size",[16,32,64,128])
     optimizer_name=trial.suggest_categorical("optimizer_name",['Adam','SGD','RMSprop'])
     weight_decay=trial.suggest_float("weight_decay",1e-5,1e-3,log=True)

     train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
     test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
     # model init
     input_dim=784
     output_dim=10
     model=MyNN(input_dim,output_dim,num_hidden_layers,num_neurons,dropout_rate)
     model.to(device) 
     loss= nn.CrossEntropyLoss()
     if optimizer_name=='Adam':
         optimizer=optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay) 
     elif optimizer_name=='SGD':
         optimizer=optim.SGD(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
     else:
         optimizer=optim.RMSprop(model.parameters(),lr=learning_rate, weight_decay=weight_decay)

     # Training Loop
     for epoch in range(epochs):
      for batch_features, batch_labels in train_loader:
          batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
          out=model(batch_features) # forward pass
          error=loss(out, batch_labels) # Loss calculation
          optimizer.zero_grad() # clearing gradient
          error.backward() # backward pass
          optimizer.step() # gradient update
     # Model Evaluation
     model.eval() 
     # Evaluation
     # model evaluation mode
     total=0
     correct=0
     with torch.no_grad():
      for batch_features, batch_labels in test_loader:
          batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
          output=model(batch_features)
          _, pred=torch.max(output,1)
          total=total+batch_labels.shape[0]
          correct=correct+(pred==batch_labels).sum().item()
     accuracy=correct/total
     return accuracy

# Creating a study using optuna
study=optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=10)
print(study.best_value)
print(study.best_params)





# # Evaluation Training data
# model.eval() # model evaluation mode
# total=0
# correct=0
# with torch.no_grad():
#      for batch_features, batch_labels in train_loader:
#           batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#           output=model(batch_features)
#           _, pred=torch.max(output,1)
#           total=total+batch_labels.shape[0]
#           correct=correct+(pred==batch_labels).sum().item()
# print(correct/total)



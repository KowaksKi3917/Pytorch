import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
# Seeding
torch.manual_seed(42)
df=pd.read_csv("/Users/parikshitbhardwaj/Downloads/fmnist_small.csv")
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
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False)
# Neural Netweork
class ANN(nn.Module):
     def __init__(self, num_features):
          super().__init__()
          self.model=nn.Sequential(
               nn.Linear(num_features, 128),
               nn.ReLU(),
               nn.Linear(128,64),
               nn.ReLU(),
               nn.Linear(64,10)
          )
     def forward(self,x):
          return self.model(x)
     
learning_rate=0.1
epochs=100
model=ANN(X_train.shape[1]) # Teling model the no of features
loss= nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
     total_loss=0
     for batch_features, batch_labels in train_loader:
          out=model(batch_features) # forward pass
          error=loss(out, batch_labels) # Loss calculation
          optimizer.zero_grad() # clearing gradient
          error.backward() # backward pass
          optimizer.step() # gradient update
          total_loss= total_loss+error.item()
     avg_loss=total_loss/len(train_loader)
     print(f"Epoch : {epoch+1} , Loss : {avg_loss:.4f}")

# Evaluation
model.eval() # model evaluation mode
total=0
correct=0
with torch.no_grad():
     for batch_features, batch_labels in test_loader:
          output=model(batch_features)
          _, pred=torch.max(output,1)
          total=total+batch_labels.shape[0]
          correct=correct+(pred==batch_labels).sum().item()
print(correct/total)



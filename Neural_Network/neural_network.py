import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


if(torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

data_folder = os.path.join("clean_data")

csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
                               
print(f"Loaded {len(csv_files)} files with {len(df)} rows total")
df['C_IV_PREV'] = df['C_IV'].shift(1)
df['C_IV_ROLL_MEAN_5'] = df['C_IV'].rolling(window=5).mean()
df = df.replace([float('inf'), -float('inf')], 0)  
df = df.dropna()   
print(df.head())
print(df.shape)

inputs = [
    'UNDERLYING_LAST', 'DTE', 'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 
    'C_RHO', 'C_VOLUME', 'C_LAST', 'C_BID', 'C_ASK', 'STRIKE',
    'P_BID', 'P_ASK', 'P_LAST', 'P_DELTA', 'P_GAMMA', 'P_VEGA', 'P_THETA', 
    'P_RHO', 'P_IV', 'P_VOLUME', 'STRIKE_DISTANCE', 'STRIKE_DISTANCE_PCT', 'C_IV_PREV',
    'C_IV_ROLL_MEAN_5',
    ]

target = 'C_IV'
scaler_x = StandardScaler()

X = df[inputs].values
Y = df[target].values.reshape(-1, 1) #this makes the Y tensor be 2-dimensional so it has the same shape as the X tensor

X = scaler_x.fit_transform(X)
xtensor = torch.tensor(X, dtype = torch.float32).to(device)
ytensor = torch.tensor(Y, dtype = torch.float32).to(device)

#this model is the function that computes the implied volatility from the inputs
#There are 4 layers to the model
#inbetween the linear layers, a rectified linear unit is applied to the previous output, which converts negative values
# in the matricies to 0, introducing non-linearity which allows for the model to represent more complex functions
model = nn.Sequential(nn.Linear(len(inputs), 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32,1)).to(device)

loss_function = nn.MSELoss()                           #mean-squared error
optimizer = optim.SGD(model.parameters(), lr = 0.05)       #optimization mechanism, stochastic gradient descent, can experiment with learning rate later

dataset = TensorDataset(xtensor, ytensor)
#loader = DataLoader(dataset, batch_size = 1000, shuffle=True)    # this is from when I was using batching
num_epochs = 1000     #number of learning cycles, can vary this in the future

for epoch in range(num_epochs):
    
    optimizer.zero_grad()                                 #clear old gradients
    loss = loss_function(model(xtensor), ytensor)       #calculate the loss at this step
    loss.backward()                                      #use back-propogation to calculate gradients for all model weights
    optimizer.step()                                    #adjusts the weights using these gradients
    if(epoch % 50 == 0):
        print(f"Epoch {epoch}/{num_epochs} Loss:{loss.item()}")
    

    """
    #I decided to experiment with batching to make the model faster, 
    #but with GPU acceleration batching was no longer faster
    total_loss = 0
    for xbatch, ybatch in loader:        
        xbatch = xbatch.to(device)
        ybatch = ybatch.to(device)
        optimizer.zero_grad()                         
        loss = loss_function(model(xbatch), ybatch)    
        loss.backward()                             
        optimizer.step()                    
        total_loss += loss.item()
    total_loss/=len(loader)
    print(f"Epoch {epoch}/{num_epochs} Loss:{total_loss}")
    """

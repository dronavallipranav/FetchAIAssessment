import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

"""
ShallowNet is a simple shallow neural network with one hidden layer followed by a ReLU activation function. 
It takes in 4 inputs and outputs a single value.
"""
class ShallowNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #ReLU is the activation function
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#Data Preprocessing
df=pd.read_csv("data_daily.csv")
df['day_number'] = np.arange(len(df))
df['month'] = df['# Date'].str[5:7]
df['day'] = df['# Date'].str[8:].astype('int32')
#Adding lag values
df['lag_1'] = df['Receipt_Count'].shift(1)
df['lag_2'] = df['Receipt_Count'].shift(2)
df = df.dropna() 

X = df[['day_number', 'day', 'lag_1', 'lag_2']].values.astype('float32')
print(X)
y = df['Receipt_Count'].values.astype('float32')

#using sklearn for efficent testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Convert to tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

input_size = X_train.shape[1]
model = ShallowNet(input_size, 5, 1)
criterion = nn.MSELoss()
torch.manual_seed(43) 
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 3000
for epoch in range(epochs):

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    #Gradient is reset
    optimizer.zero_grad()
    #Gradient is recalculated
    loss.backward()
    #Weights are updated
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    mse = criterion(y_pred, y_test_tensor)
    print('Final Mean Squared Error on Test Data:', mse.item())
    rse = np.sqrt(mse.item())
    print(rse)
    
torch.save(model.state_dict(), 'model.pth')
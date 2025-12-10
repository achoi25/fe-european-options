import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

if(torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

calls_folder = "clean_data_2/calls"
puts_folder = "clean_data_2/puts"

call_files = sorted(glob.glob(f"{calls_folder}/*.pkl"))
put_files  = sorted(glob.glob(f"{puts_folder}/*.pkl"))

inputs = [
    'UNDERLYING_LAST', 'DTE', 'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 
    'C_RHO', 'C_VOLUME', 'C_LAST', 'C_BID', 'C_ASK', 'STRIKE',
     'STRIKE_DISTANCE', 'STRIKE_DISTANCE_PCT', 
    'C_IV_PREV', 'C_IV_ROLL_MEAN_5'
]

target = 'C_IV'

# Store validation metrics per month
monthly_metrics = []

for month_idx, (call_file, put_file) in enumerate(zip(call_files, put_files)):
    print(f"\nProcessing Month {month_idx+1}: {call_file.split('/')[-1]}:")

    # load this months data
    df_calls = pd.read_pickle(call_file)
    df_puts = pd.read_pickle(put_file)
    df_month = pd.concat([df_calls, df_puts], ignore_index=True)

    #computing strike distance and strike distance percentage
    df_month['STRIKE_DISTANCE'] = df_month['STRIKE'] - df_month['UNDERLYING_LAST']
    df_month['STRIKE_DISTANCE_PCT'] = df_month['STRIKE_DISTANCE'] / df_month['UNDERLYING_LAST']


    #computing previous C_IV and 5 day moving average
    df_month = df_month.sort_values('QUOTE_UNIXTIME')
    df_month['C_IV_PREV'] = df_month['C_IV'].shift(1)
    df_month['C_IV_ROLL_MEAN_5'] = df_month['C_IV'].rolling(5).mean()

    df_month = df_month.dropna(subset=['C_IV_PREV', 'C_IV_ROLL_MEAN_5'])
    df_month = df_month[df_month['C_IV'] < 1]     #filtering out unrealistic implied volatility values greater than 1

    df_month[inputs] = df_month[inputs].fillna(0)     #filling any remaning nans with 0, because nans would propogate throught the layers and corrupt the model

    #20% of the data will be used for validating the model each month
    val_size = int(len(df_month) * 0.2)
    training_data = df_month[:-val_size]
    validation_data = df_month[-val_size:]


    scaler_x = StandardScaler()
    X_train = training_data[inputs].values
    Y_train = training_data[target].values.reshape(-1,1) #this makes the Y tensor be 2-dimensional so it has the same shape as the X tensor
    X_train_scaled = scaler_x.fit_transform(X_train)

    X_val = validation_data[inputs].values
    Y_val = validation_data[target].values.reshape(-1,1)
    X_val_scaled = scaler_x.transform(X_val)

    
    xtensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    ytensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)

    dataset = TensorDataset(xtensor, ytensor)
    loader = DataLoader(dataset, batch_size=100000, shuffle=True)

    #this model is the function that computes the implied volatility from the inputs
    #There are 5 layers to the model
    #inbetween the linear layers, a rectified linear unit is applied to the previous output, which converts negative values
    # in the matricies to 0, introducing non-linearity which allows for the model to represent more complex functions
    model = nn.Sequential(nn.Linear(len(inputs), 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32,1)).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.05)
    loss_function = nn.MSELoss()

    # Training for 500 epochs
    num_epochs = 500
    for epoch in range(num_epochs):
        total_loss = 0
        for xbatch, ybatch in loader:
            optimizer.zero_grad()
            loss = loss_function(model(xbatch), ybatch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(loader)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss:.6f}")

    #validating the model
    model.eval()
    with torch.no_grad():
        Y_val_pred = model(X_val_tensor).cpu().numpy()
    rmse_val = np.sqrt(mean_squared_error(Y_val, Y_val_pred))
    r2 = r2_score(Y_val, Y_val_pred)
    print(f"Validation RMSE: {rmse_val:.6f}, R^2: {r2:.6f}")

    monthly_metrics.append({'month': call_file.split('_')[-1].split('.')[0], 'rmse': rmse_val,'r2': r2})

    """
    #This code is to show plots of the predicted vs actual implied volatility for each month
    #uncomment to show, but it does increase runtime further
    plt.figure(figsize=(6,6))
    plt.scatter(Y_val, Y_val_pred, s=2)
    plt.xlabel("Actual C_IV")
    plt.ylabel("Predicted C_IV")
    month_label = call_file.split('_')[-1].split('.')[0]
    plt.title(f"Predicted vs Actual C_IV - {month_label}")
    plt.tight_layout()
    plt.show()
    """

# Extract all R^2 values
r2_values = [m['r2'] for m in monthly_metrics]

# Compute summary statistics
avg_r2 = np.mean(r2_values)
max_r2 = np.max(r2_values)
min_r2 = np.min(r2_values)

print("Monthly Validation R^2 Summary: ")
print(f"Average R^2 across months: {avg_r2:.6f}")
print(f"Highest R^2 reached:       {max_r2:.6f}")
print(f"Lowest R^2 reached:        {min_r2:.6f}")

best_month = monthly_metrics[np.argmax(r2_values)]['month']
worst_month = monthly_metrics[np.argmin(r2_values)]['month']

print(f"Best month:  {best_month}")
print(f"Worst month: {worst_month}")

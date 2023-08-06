import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from Actor_Critic_LSTM_NN import LSTM_model
from torch.utils.data import DataLoader
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import os
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def create_sequences(data, seq_length):
    data = data.reset_index(drop=True)
    xs = []
    ys = []
    for i in (range(len(data) - seq_length)):
        x = data[i:(i + seq_length)]
        y = data['Close'][i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)



def LSTM_training(ticker, hidden, rolling_window, epochs, LR):
    loss = nn.MSELoss()
    scaler = MinMaxScaler()

    ticker_data = yf.download(ticker)
    ticker_data_fixed = ticker_data.tail(3861)
    ticker_data_fixed_reset = ticker_data_fixed.reset_index(drop=True)
    ticker_data_fixed_mod = ticker_data_fixed_reset[['Open', 'High', 'Low', 'Close', 'Volume']]

    norm_ticker_data = pd.DataFrame(scaler.fit_transform(ticker_data_fixed_mod), columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    training_size = round(3861 * 0.8)
    testing_size = round(3861 * 0.2)

    Training_data = norm_ticker_data.head(training_size)
    Testing_data = norm_ticker_data.tail(testing_size)

    model = LSTM_model(5,hidden, 2, 1)

    xs, ys = create_sequences(Training_data, rolling_window)
    dataset = TimeSeriesDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    xs, ys = create_sequences(Testing_data, rolling_window)
    test_dataset = TimeSeriesDataset(xs, ys)
    real_data = ys
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    for ep in range(epochs):
        model.train()
        for batch in dataloader:
            x , y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)

            Loss = loss(pred.unsqueeze(1), y.unsqueeze(1))
            Loss.backward()
            optimizer.step()


    print(f"Epoch {ep+1}, Loss: {Loss} ")

    predizioni = []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            x_test , y_test = batch
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            pred_test = model(x_test)

            test_loss = loss(pred_test.unsqueeze(1), y_test.unsqueeze(1))
            predizioni.append(pred_test.cpu())
        print(f"Epoch {ep+1}, Test Loss: {test_loss}")

    dati_pred = torch.cat(predizioni, dim=0).numpy()
    real_data = real_data

    directory = f'best models/{ticker}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    time = np.arange(len(dati_pred))
    plt.figure(figsize=(10, 5))

    plt.plot(time, real_data, label='Real Data')
    plt.plot(time, dati_pred, label='Predicted Data', linestyle='--')

    plt.title('Real vs Predicted Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig(f"{directory}/image_{hidden}_{rolling_window}_{epochs}_{LR}.png")
    plt.close()


    torch.save(model.state_dict(), f"best models/best_{ticker}_model.pth")
    print("model saved")

    return Loss, test_loss






ASSETS = {
    0   : ["7203.T", "9984.T", "6758.T", "6861.T", "9983.T"],
    1   : ["BRK-A", "JNJ", "PG", "V", "JPM"],
    2   : ["AAPL", "AMZN", "MSFT", "GOOGL", "NVDA"],
    3   : ["ULVR.L", "HSBA.L", "BATS.L", "DGE.L", "BP.L"]
}

"""
print("training best models")

LSTM_training("V", 300, 10, 60, 0.0001)

"""


"""
def grid_search(params_grid, tick):
    keys = params_grid.keys()
    values = params_grid.values()
    results = []
    for combination in product(*values):
        params_combination = dict(zip(keys, combination))
        print(f"Training with parameters: {params_combination}")
        Loss, test_loss, acc = LSTM_training(tick,
                                        params_combination['hidden'],
                                        params_combination['output'],
                                        params_combination['rolling_window'],
                                        params_combination['epochs'],
                                        params_combination['LR'])
        params_combination['training_loss'] = Loss.item()
        params_combination['test_loss'] = test_loss.item()
        params_combination['accuracy'] = acc
        results.append(params_combination)

    results_df = pd.DataFrame(results)
    results_df.to_excel(f'grid_search_results_{tick}.xlsx', index=False)

"""

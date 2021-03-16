'''
script for LSTM demo
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler


'''
https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
'''

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        lstm_output, (h, c) = self.lstm(x, hidden)
        lstm_output = lstm_output.view(-1, self.hidden_dim)
        model_output = self.fc(lstm_output)
        return model_output, (h, c)

torch.manual_seed(1)
torch_lstm = LSTM(input_size=2, output_size=1, hidden_dim=3)

data = [[1, 1],
        [2, 2],
        [3, 3]]
torch_batch = torch.Tensor(data).unsqueeze(0)
torch_output, (h, c) = torch_lstm(torch_batch, None)
print()
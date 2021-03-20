# '''
# script for LSTM demo
# '''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer



# '''
# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# https://cnvrg.io/pytorch-lstm/
# https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py
# https://github.com/Ferdib-Al-Islam/lstm-time-series-prediction-pytorch/blob/master/lstm-time-series-pytorch.ipynb
# https://github.com/hunkim/PyTorchZeroToAll
# https://towardsdatascience.com/tutorial-on-lstm-a-computational-perspective-f3417442c2cd
# https://www.kaggle.com/akhileshrai/tutorial-early-stopping-vanilla-rnn-pytorch
# '''
#
# def scaling_window(data, seq_length):
#     x = []
#     y = []
#
#     for i in range(len(data)-seq_length-1):
#         _x = data[i:(i+seq_length)]
#         _y = data[i+seq_length]
#         x.append(_x)
#         y.append(_y)
#
#     return np.array(x), np.array(y)
#
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, batch_size):
#         super(RNN, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size
#         # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         # -> x needs to be: (batch_size, seq, input_size)
#
#         # or:
#         # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         # Set initial hidden states (and cell states for LSTM)
#         h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
#
#         # x: (n, 28, 28), h0: (2, n, 128)
#
#         # Forward propagate RNN
#         # out, _ = self.rnn(x, h0)
#         # or:
#         out, _ = self.lstm(x, (h0, c0))
#
#         # out: tensor of shape (batch_size, seq_length, hidden_size)
#         # out: (n, 28, 128)
#
#         # Decode the hidden state of the last time step
#         out = out[:, -1, :]
#         # out: (n, 128)
#
#         out = self.fc(out.float())
#         # out: (n, 10)
#         return out
#
# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Hyper-parameters
# num_epochs = 100
# batch_size = 1
# learning_rate = 0.01
#
# input_size = 1
# sequence_length = 4
# hidden_size = 3
# num_layers = 1
#
#
# model = RNN(input_size, hidden_size, num_layers, batch_size).to(device)
#
# # Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# data = pd.read_csv('./airline_passengers.txt')
# train_set = data.iloc[:100, [1]]
# test_set = data.iloc[100:, [1]]
#
#
# scaler = StandardScaler()
# train_set = scaler.fit_transform(train_set)
#
# x_train, y_train = scaling_window(train_set, sequence_length)
# x_train = torch.from_numpy(x_train)
# y_train = torch.from_numpy(y_train)
#
# torch.manual_seed(1)
# losses = []
# for epoch in np.arange(num_epochs):
#     for input, output in zip(x_train, y_train):
#         input = input.unsqueeze(0).to(device)
#         outputs = model(input)
#         loss = criterion(outputs[0], output.to(device).float())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     losses.append(loss.item())
#     if epoch > 20 and np.abs(losses[-10]/losses[-1] - 1) < 1/100:
#         break
#     if epoch % 20 == 0:
#         print(f'Epoch:{epoch+1} / {num_epochs} - Loss:{loss.item():.4f}')
#
# test_set = scaler.fit_transform(test_set)
# x_test, y_test = scaling_window(test_set, sequence_length)
# x_test = torch.from_numpy(x_test)
# y_test = torch.from_numpy(y_test)
#
# # #state
# state = model.state_dict()
# rnn_ih = state['rnn.weight_ih_l0']
# rnn_hh = state['rnn.weight_hh_l0']
#
#
#
# with torch.no_grad():
#     preds = []
#     for input, output in zip(x_test, y_test):
#         input = input.unsqueeze(0).to(device)
#         pred = model(input)
#         preds.append(pred)
# preds_inv = [scaler.inverse_transform(y[0].cpu().numpy())[0] for y in preds]
# plt.plot(preds_inv)
# true_y = scaler.inverse_transform(test_set[sequence_length:])
# plt.plot(true_y)
# nrmse = (np.sqrt(np.mean((true_y - preds_inv)**2))) / np.std(true_y)
#
# print()

def calculate_nrmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2)) / np.std(true)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, n_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True,)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
       (h0, c0) = (torch.zeros(size=(self.n_layers, x.size(0), self.hidden_dim)),
                    torch.zeros(size=(self.n_layers, x.size(0), self.hidden_dim)))

       x = x.view(x.size(0), -1, self.input_dim)
       lstm_out, _ = self.lstm(x, (h0, c0))
       last_out = lstm_out[:, -1, :]
       out = self.fc(last_out)
       return out

# #model hyperparameters
sequence_len = 5
batch_size = 10
n_layers = 1
hidden_dim = 5
epochs = 500
stop_epochs = 10
lr = 1e-3
params = dict(sequence_len=sequence_len, batch_size=batch_size,
              n_layers=n_layers, hidden_dim=hidden_dim,
              epochs=epochs, lr=lr,
              stop_epochs=stop_epochs)

# #generate data
n = 1_000
xM = np.full(shape=(n, 1), fill_value=np.nan)
wM = np.random.normal(0, 1, size=(n, 1))
xM[0] = wM[0]
for t in np.arange(1, n):
    xM[t] = 0.8 * xM[t - 1] + wM[t]
# xM = np.sin(2 * np.pi * 5 * np.linspace(0, 1, n)).reshape(-1, 1)
# xM = pd.read_csv('./airline_passengers.txt')
# xM = xM['Passengers'].values.reshape(-1, 1)
# n = xM.shape[0]

# #split train-test set
train_proportion = 0.7
split_point = np.int(train_proportion * n)
train_set = xM[:split_point]
test_set = xM[split_point:]
# #normalise data
scaler = MinMaxScaler()
train_set = scaler.fit_transform(train_set)
test_set = scaler.fit_transform(test_set)
plt.plot(np.arange(split_point), train_set)
plt.plot(np.arange(split_point, n), test_set)

# #create sequences for LSTM input
X_train = []
y_train = []
for t in np.arange(sequence_len, split_point):
    temp_x = train_set[t-sequence_len:t, :]
    temp_y = train_set[t, :]
    X_train.append(temp_x)
    y_train.append(temp_y)

# #numpy to torch Tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

# #build model
model = LSTM(input_dim=sequence_len, hidden_dim=hidden_dim, batch_size=batch_size, n_layers=n_layers)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

# #train model
print('Training model', end='***')
torch.manual_seed(1)
train_losses = np.full(shape=epochs, fill_value=np.nan)
min_loss = np.inf
total_maxsteps = np.arange(0, X_train.shape[0], batch_size).size * epochs
for epoch in np.arange(epochs):
    for i in np.arange(0, X_train.shape[0], batch_size):
        if i + batch_size > X_train.shape[0]:
            break
        x_ = X_train[i:i+batch_size]
        y_ = y_train[i:i+batch_size]
    # for x_, y_ in zip(X_train, y_train):
        # #forward pass
        y_hat = model(x_)
        # #calculate loss
        loss_ = loss(y_hat, y_)

        # #backpropagate and update weights
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
    # #store epoch loss
    train_losses[epoch] = loss_.item()
    # #show current loss
    if (epoch+1) % 5 == 0:
        print(f'Epoch{epoch+1}/{epochs} - Loss:{loss_.item():.4f}')
        print(f'Step{np.arange(0, X_train.shape[0], batch_size).size * (epoch+1)}/{total_maxsteps}')
    # #early stopping implementation
    if loss_.item() < min_loss:
        min_loss = loss_.item()
        epoch_no_improve = 0
    else:
        epoch_no_improve += 1
    if epoch_no_improve > stop_epochs:
        print(f'Early Stop at epoch {epoch+1} from total {epochs}')
        break

print('Model trained')
fig, ax = plt.subplots(1, 1)
ax.plot(np.arange(epochs), train_losses, marker='x', alpha=0.6)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Epochs VS Train Loss')

# #Out of sample test
# #create sequences for LSTM input
X_test = []
y_test = []
for t in np.arange(sequence_len, n-split_point):
    temp_x = test_set[t - sequence_len:t, :]
    temp_y = test_set[t, :]
    X_test.append(temp_x)
    y_test.append(temp_y)

# #numpy to torch Tensors
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# #evaluate on whole time series
with torch.no_grad():
    predictions = []
    for x_  in torch.cat([X_train, X_test]):
        x_ = x_.unsqueeze(0)
        y_hat = model(x_)
        predictions.append(y_hat.numpy()[0])
predictions = np.array(predictions)

# #calculate in-sample --- out-of-sample nrmse
nrmse_insample = calculate_nrmse(true=y_train.numpy(), pred=predictions[:X_train.shape[0]])
nrmse_outsample = calculate_nrmse(true=y_test.numpy(), pred=predictions[X_train.shape[0]:])

# #plot results
# #whole sample
fig, ax = plt.subplots(3, 1)
plt.suptitle(f'LSTM')
ax[0].plot(predictions, label='pred', alpha=0.8, linestyle='--')
ax[0].axvline(split_point, color='red', linestyle='-')
ax[0].plot(torch.cat([y_train, y_test]).numpy(), label='true', alpha=0.5)
ax[0].legend()
ax[0].set_title('True - Predicted')
# #in sample
ax[1].plot(predictions[:X_train.shape[0]], label='pred', alpha=0.8, linestyle='--')
ax[1].plot(y_train.numpy(), label='true', alpha=0.5)
ax[1].legend()
ax[1].set_title('In sample (fit-train model)')
# #out of sample
ax[2].plot(predictions[X_train.shape[0]:], label='pred', alpha=0.8, linestyle='--')
ax[2].plot(y_test.numpy(), label='true', alpha=0.5)
ax[2].legend()
ax[2].set_title('Out if sample (fit-test model)')
plt.show()
print(f'Params:{params}', end='\n')
print(f'Nrmse:{nrmse_insample} - {nrmse_outsample}', end='\n')
print()


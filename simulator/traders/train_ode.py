import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy

import seaborn as sns
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import os
import multiprocessing

from torchdyn.models import *
from torchdyn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class timeseries(Dataset):
    '''Sequenced tick-by-tick LOB dataset

    Args:
        X: Sequenced LOB data.
        y: Labels for every sequence.
        with_time: If set to true then timestamps are included, if false then
        they are omitted.
    Returns:
        LOB sequenced dataset (subtract 1 from each index if with_time==False):
        0: timestamp in seconds (if with_time set to true)
        1: best bid price
        2: best ask price
        3: ask volume
        4: bid volume
        5: microprice
        6: exponential moving average of microprice
        7: delta-mu, change in microprice from this timestep to the last one
        in the sequence
    '''
    def __init__(self,x,y,with_time=True):
        if with_time:
            self.x = torch.tensor(x,dtype=torch.float32)[:,:,[0,1,2,3,4,5,6,9]]
        else:
            self.x = torch.tensor(x,dtype=torch.float32)[:,:,[1,2,3,4,5,6,9]]

        print(self.x.shape)
        self.y = torch.tensor(y,dtype=torch.long)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.len

class GBPUSDDataModule(pl.LightningDataModule):
    def __init__(self, data_type='GBPUSD', window=10, batch_size=1, pred_horizon=10, alpha=0.0002, with_time=True):
        super().__init__()
        self.window = window
        self.batch_size = batch_size
        self.pred_horizon = pred_horizon
        self.with_time = with_time
        self.data_type = data_type
        self.alpha = alpha

    def setup(self, stage=None):
        train_dfs, test_dfs = self.load_data(train_days=8, test_days=1)
        X_train, y_train = self.process_data(train_dfs)
        X_test, y_test = self.process_data(test_dfs)

        self.train_data = timeseries(X_train, y_train, with_time=self.with_time)
        self.test_data = timeseries(X_test, y_test, with_time=self.with_time)

    def load_data(self, train_days, test_days):
        path_to_data = "C:/Users/Angus Parsonson/Documents/University/FourthYear/Diss/Dissertation/data/" + self.data_type + "/"
        train_dfs = []
        test_dfs = []
        for i, f in enumerate(os.listdir(path_to_data)):
            print(path_to_data + f)
            df = pd.read_csv(path_to_data + f)
            if (i < train_days):
                train_dfs.append(df)
            elif (i < train_days+test_days):
                test_dfs.append(df)
            else:
              break

        return (train_dfs, test_dfs)

    def process_data(self, dfs):
        df = self.normalise(np.array(self.convert_to_seconds(dfs[0])))
        X, y = self.sequence_data(df, self.window)
        for i in range(1, len(dfs)):
            new_df = self.normalise(np.array(self.convert_to_seconds(dfs[i])))
            new_X, new_y = self.sequence_data(new_df, self.window)
            X = np.concatenate((X, new_X), axis=0)
            y = np.concatenate((y, new_y), axis=0)

        return X, y

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)#, num_workers=8)

        return train_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)#, num_workers=8)

        return test_dataloader

    def normalise(self, data):
        ind = [1,2,3,4,5,6,8]
        mc = data[:,ind]
        data[:,ind] = (mc - mc.min()) / (mc.max() - mc.min())

        return data

    def convert_to_seconds(self, df):
        df = self.convert_to_microprice(df, moving_avg_size=self.pred_horizon)
        time_in_seconds = []
        start = 0.0
        for index, row in df.iterrows():
            tokens = row['Local time'].split()
            time = tokens[1]
            h, m, s = time.split(':')
            time = float(int(h) * 3600 + int(m) * 60 + float(s))
            if (index == 0):
                start = time
            time_in_seconds.append([row['Microprice'], time - start])
        time_in_seconds = np.array(time_in_seconds)
        # plt.plot(time_in_seconds[:,1], time_in_seconds[:,0])
        # plt.show()
        df['Local time'] = time_in_seconds[:,1]

        return df

    def convert_to_microprice(self, df, moving_avg_size=5, smoothing=2):
        df['Microprice'] = (df['Bid']*df['AskVolume'] + df['Ask']*df['BidVolume']) / (df['AskVolume'] + df['BidVolume'])
        MicroExpMovingAvg = []
        mult = smoothing / (1.0 + moving_avg_size)
        sma = 0.0
        for i in range(len(df)):
            if (i < moving_avg_size):
                sma += df.iloc[i]['Microprice']
                MicroExpMovingAvg.append(sma/float(i+1))
            else:
                MicroExpMovingAvg.append((df.iloc[i]['Microprice'] * mult) + (MicroExpMovingAvg[i-1]) * (1.0 - mult))

        df['NormMovingAvg'] = MicroExpMovingAvg
        df['MicroExpMovingAvg'] = MicroExpMovingAvg
        df['OrderBookImbalance'] = (df['BidVolume'] - df['AskVolume']) / (df['BidVolume'] + df['AskVolume'])
        return df

    def get_dir(self, curr_mvavg, next_mvavg):
        lt = (next_mvavg- curr_mvavg) / curr_mvavg
        if (lt > self.alpha):
            direction = 2
        elif (lt < -self.alpha):
            direction = 0
        else:
            direction = 1

        return direction

    def sequence_data(self, data, window):
        X = []
        Y = []
        stag = 0
        up = 0
        dwn = 0
        L = []
        for i in range(0, len(data)-window):
            j = i+window-1
            next_dir_idx = j+self.pred_horizon
            # curr_dir_idx = i-1
            curr_dir_idx = j-self.pred_horizon
            curr_mvavg = data[j][7]
            curr_time = data[j][0]

            curr_dir = 0
            if (next_dir_idx < len(data)):
                if (curr_dir_idx >= 0):
                    curr_dir = self.get_dir(data[curr_dir_idx][7], curr_mvavg)
                    # curr_dir = Y[curr_dir_idx]
                label = self.get_dir(curr_mvavg, data[next_dir_idx][7])

                if (label == 2):
                    up += 1
                elif (label == 0):
                    dwn += 1
                else:
                    stag += 1

                # print(i, j, curr_dir_idx, data[i], data[j], label, prev_label, len(Y))
                L.append([curr_time, label])
                seq = data[i:i+window]
                inputs_seq = []
                for k in range(0, len(seq)):
                    inputs_seq.append(np.append(seq[k], data[j][7]-seq[k][7]))

                X.append(inputs_seq)
                Y.append(label)

        prev_lab = 0
        prev_time = data[0][0]
        shades = []
        for l in L:
            if (prev_lab != l[1]):
                shades.append([prev_time, l[0], prev_lab])
                prev_time = l[0]
                prev_lab = l[1]

        for shade in shades:
            if shade[2] == 0:
                col = 'r'
            elif shade[2] == 1:
                col = 'b'
            else:
                col = 'g'

            # plt.axvspan(shade[0], shade[1], color=col, alpha=0.5, lw=0)
        # plt.plot(data[:,0], data[:,1])

        # plt.show()
        # np.set_printoptions(threshold=sys.maxsize)
        # print(np.array(X[:self.pred_horizon]))
        # print("up: " + str(up) + " , down: " + str(dwn) + ", stagnant: " + str(stag))
        print("Labels Done")
        return np.array(X), np.array(Y)

class LSTM(pl.LightningModule):
    def __init__(self, input_size=2, hidden_size=100, seq_len=10, num_layers=1, batch_size=1, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, output_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm1d(self.seq_len)

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(self.bn(x))
        # lstm_out, self.hidden = self.lstm(x)
        # lstm_out = self.dropout(lstm_out[:,-1,:])
        lstm_out = lstm_out[:,-1,:]
        lstm_out = self.relu(self.fc1(lstm_out))
        predictions = self.fc2(lstm_out)

        return predictions

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.logger.experiment.add_scalar("Loss/Train", loss)

        return loss

    def test_step(self, batch, batch_idx):
        X, labels = batch
        logits = self.forward(X)
        # print(logits)
        total = 0
        correct = 0
        for i in range(len(logits)):
            # pred = logits[i].argmax(dim=0, keepdim=True)
            pred = torch.max(logits[i], dim = 1)
            # print(logits[i], labels[i])
            if (pred[0] == labels[i]):
                correct += 1
            total += 1

        metrics = {'correct': correct, 'total': total}
        return metrics

    def test_epoch_end(self, outputs):
        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        print(100*correct/total)
        return {'overall_accuracy': 100*correct/total}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

class ODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_ODE=True):
        super(ODELSTMCell, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_ODE = use_ODE

        self.ode = NeuralDE(self.fc, solver='rk4', sensitivity='autograd')

    def forward(self, inputs, hx, ts):
        batch_size = ts.size(0)

        # new_h = torch.zeros(batch_size, hx[0].size(1)).to(device)
        if (self.use_ODE):
            new_h, new_c = self.lstm(inputs.to(device), (hx[0].to(device), hx[1].to(device)))
            new_h, new_c = self.lstm(inputs.to(device), (new_h.to(device), new_c.to(device)))
            ht = torch.zeros(batch_size, hx[0].size(1)).to(device)
            # ht = self.ode.trajectory(new_h, ts[0])[1]
            for batch_idx, batch in enumerate(ts):
                ht[batch_idx] = self.ode.trajectory(new_h[batch_idx].to(device), batch.to(device))[1].to(device)

            return (ht, new_c)
            # for batch_idx, batch in enumerate(ts):
            #     new_h[batch_idx] = self.ode.trajectory(hx[0][batch_idx].to(device), batch.to(device))[1].to(device)

            # new_h, new_c = self.lstm(inputs.to(device), (new_h.to(device), hx[1].to(device)))
            # new_h, new_c = self.lstm(inputs.to(device), (new_h.to(device), new_c.to(device)))

        new_h, new_c = self.lstm(inputs.to(device), (hx[0].to(device), hx[1].to(device)))
        new_h, new_c = self.lstm(inputs.to(device), (new_h.to(device), new_c.to(device)))

        return (new_h, new_c)

class ODELSTM(pl.LightningModule):
    def __init__(self, use_ODE=True, input_size=2, hidden_size=100, seq_len=10, output_size=3):
        super(ODELSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = ODELSTMCell(input_size, hidden_size, use_ODE=use_ODE)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, output_size)
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(self.seq_len)

    def forward(self, x, timespans):
        batch_size = x.size(0)
        seq_len = x.size(1)

        hidden_state = (
            torch.zeros((batch_size, self.hidden_size)),
            torch.zeros((batch_size, self.hidden_size)),
        )

        outputs = []
        x = self.bn(x)
        for t in range(1, seq_len):
            inputs = x[:,t]
            ts = timespans[:,t-1:t+1]
            hidden_state = self.cell.forward(inputs, hidden_state, ts)

        lstm_out = self.dropout(hidden_state[0])
        lstm_out = self.relu(self.fc1(lstm_out))
        predictions = self.fc2(lstm_out)
        # outputs = torch.stack(outputs, dim=1)

        return predictions

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X[:,:,list(range(1,self.input_size+1))], X[:,:,0])
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.logger.experiment.add_scalar("Loss/Train", loss)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X[:,:,list(range(1,self.input_size+1))], X[:,:,0])
        total = 0
        correct = 0
        true_pos_up = 0
        true_pos_dwn = 0
        false_pos_up = 0
        false_pos_dwn = 0
        false_neg_dwn = 0
        false_neg_up = 0
        for i in range(len(logits)):
            pred = logits[i].argmax(dim=0, keepdim=True)
            # pred = torch.max(logits[i], dim = 0)
            # print(pred)
            if (pred[0] == y[i]):
                correct += 1
                if (y[i] == 2): 
                    true_pos_up += 1
                else:
                    true_pos_dwn += 1
            else: 
                if (pred[0] == 2): 
                    false_pos_up += 1
                    false_neg_dwn += 1
                else:
                    false_pos_dwn += 1
                    false_neg_up += 1

            total += 1

        # precision, recall, F1
        metrics = { 'correct': correct, 
                    'total': total,
                    'true_pos_up': true_pos_up,
                    'true_pos_dwn': true_pos_dwn,
                    'false_pos_up': false_pos_up,
                    'false_pos_dwn': false_pos_dwn,
                    'false_neg_up': false_neg_up,
                    'false_neg_dwn': false_neg_dwn}

        return metrics

    def test_epoch_end(self, outputs):
        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        true_pos_up = sum([x['true_pos_up'] for x in outputs])
        false_pos_up = sum([x['false_pos_up'] for x in outputs])
        true_pos_dwn = sum([x['true_pos_dwn'] for x in outputs])
        false_pos_dwn = sum([x['false_pos_dwn'] for x in outputs])
        false_neg_dwn = sum([x['false_neg_dwn'] for x in outputs])
        false_neg_up = sum([x['false_neg_up'] for x in outputs])

        precision_up = true_pos_up / (true_pos_up + false_pos_up)
        precision_dwn = true_pos_dwn / (true_pos_dwn + false_pos_dwn)
        recall_up = true_pos_up / (true_pos_up + false_neg_up)
        recall_dwn = true_pos_dwn / (true_pos_dwn + false_neg_dwn)
        precision = (precision_up + precision_dwn) / 2
        recall = (recall_up + recall_dwn) / 2
        print("Accuracy: " + str(100*correct/total))
        print("Precision up: " + str(precision_up))
        print("Precision down: " + str(precision_dwn))
        print("Recall up: " + str(recall_up))
        print("Recall down: " + str(recall_dwn))
        print("F1-score: " + str(2 * (precision * recall) / (precision + recall)))

        # self.log({'overall_accuracy': 100*correct/total,
        #         'up_precision': precision_up,
        #         'down_precision': precision_dwn,
        #         'up_recall': recall_up,
        #         'down_recall': recall_dwn,
        #         'precision': precision,
        #         'recall': recall,
        #         'F1-score': 2 * (precision * recall) / (precision + recall)})
        return {'overall_accuracy': 100*correct/total}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == '__main__':
    # print(torch.cuda.is_available())
    print(os.getcwd())
    data_module = GBPUSDDataModule(data_type="WTB", window=50, batch_size=64, pred_horizon=50, alpha=0.0000, with_time=True)
    # model = LSTM(input_size=7, hidden_size=100, seq_len=50, num_layers=10)
    model = ODELSTM(use_ODE=False, input_size=7, hidden_size=100, seq_len=50)
    logger = TensorBoardLogger('tb_logs', name='ode_logs')
    trainer = pl.Trainer(max_epochs=1, logger=logger)

    trainer.fit(model, data_module)
    trainer.test()

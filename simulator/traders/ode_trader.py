from traders.trader import Trader
import collections 
import random
import sys
import time

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ode = NeuralDE(self.fc, solver='rk4', sensitivity='autograd')

    def forward(self, inputs, hx, ts):
        batch_size = ts.size(0)

        # new_h = torch.zeros(batch_size, hx[0].size(1)).to(device)
        if (self.use_ODE):
            new_h, new_c = self.lstm(inputs.to(self.device), (hx[0].to(self.device), hx[1].to(self.device)))
            new_h, new_c = self.lstm(inputs.to(self.device), (new_h.to(self.device), new_c.to(self.device)))
            ht = torch.zeros(batch_size, hx[0].size(1)).to(self.device)
            # ht = self.ode.trajectory(new_h, ts[0])[1]
            for batch_idx, batch in enumerate(ts):
                ht[batch_idx] = self.ode.trajectory(new_h[batch_idx].to(self.device), batch.to(self.device))[1].to(self.device)

            return (ht, new_c)
            # for batch_idx, batch in enumerate(ts):
            #     new_h[batch_idx] = self.ode.trajectory(hx[0][batch_idx].to(device), batch.to(device))[1].to(device)

            # new_h, new_c = self.lstm(inputs.to(device), (new_h.to(device), hx[1].to(device)))
            # new_h, new_c = self.lstm(inputs.to(device), (new_h.to(device), new_c.to(device)))

        new_h, new_c = self.lstm(inputs.to(self.device), (hx[0].to(self.device), hx[1].to(self.device)))
        new_h, new_c = self.lstm(inputs.to(self.device), (new_h.to(self.device), new_c.to(self.device)))

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

        return metrics

    def test_epoch_end(self, outputs):
        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        
        return {'overall_accuracy': 100*correct/total}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

class ODETrader(Trader):
    def __init__(self, init_balance=0, pred_horizon=100):
        self.balance = init_balance
        self.pred_horizon = pred_horizon
        self.stock = 0
        self.potential_stock = 0
        self.tick_num = 0
        self.orders = []
        self.model= ODELSTM.load_from_checkpoint(__file__ + "\\..\\" + "ode_pred_20.ckpt", use_ODE=True, input_size=7, seq_len=50)
        self.model.eval()
        self.ticks = collections.deque()
        self.micros = collections.deque()
        self.emas = collections.deque()
        self.total = 0
        self.buy_orders = collections.deque()
        self.sell_orders = collections.deque()
        self.ema = 0
        self.ones = 0
        self.twos = 0
        self.diffs = []
        self.smooth = 2/(self.pred_horizon+1.0)
        self.bid_norm_vals = {'max_p': 0, 'min_p': float('inf'), 
                              'max_v': 0, 'min_v': float('inf')}
        self.ask_norm_vals = {'max_p': 0, 'min_p': float('inf'), 
                              'max_v': 0, 'min_v': float('inf')}
        self.mic_norm_vals = {'max_p': 0, 'min_p': float('inf')}
        self.ema_norm_vals = {'max_p': 0, 'min_p': float('inf')}
        self.ready = False
        self.eval_times = [0.0, 0.0]

    def respond(self, tick):
        new_orders = []
        # if self.tick_num % 100 == 0:
        #     print("time: " + str(tick['Local time']))
        self.tick_num += 1
        microprice = ((tick['Bid'] * tick['AskVolume'] + 
                    tick['Ask'] * tick['BidVolume']) / 
                    (tick['AskVolume'] + tick['BidVolume']))
        for order in list(self.buy_orders):
            if (order[0] == self.tick_num): #or (tick['Local time'] > 30000 and self.stock > 0)
                self.diffs.append(-(microprice - order[1]))
                new_orders.append({'type': 'BID', 'price': microprice, 'quantity': tick['AskVolume']})
                self.buy_orders.popleft()
            else: 
                break
        for order in list(self.sell_orders):
            if (order[0] == self.tick_num): #or (tick['Local time'] > 30000 and self.stock > 0)
                self.diffs.append(microprice - order[1])
                new_orders.append({'type': 'ASK', 'price': microprice, 'quantity': tick['BidVolume']})
                self.sell_orders.popleft()
            else: 
                break
        
        self.ticks.append(tick)
        self.micros.append(microprice)
        self.total += microprice

        if (len(self.ticks) <= 100):
            self.emas.append((self.total + microprice) / (len(self.ticks)+1))
            self.__update_norm_vals__()
        else:
            self.emas.append((microprice * self.smooth) + (self.emas[-1]*(1-self.smooth)))
            self.micros.popleft()
            self.emas.popleft()
            self.ticks.popleft()
            self.__update_norm_vals__()
            inputs, timespans = self.__gen_inputs__()

            eval_start = time.time()
            output = self.model(inputs, timespans)
            eval_end = time.time()
            self.eval_times[0] += (eval_end - eval_start)
            self.eval_times[1] += 1
            # print(self.eval_times[0] / self.eval_times[1])

            direction = torch.argmax(output).item()
            certainty = torch.exp(output)[0][direction].item() / sum(torch.exp(output)[0])

            if (tick['Local time'] <= 28000 and certainty >= 0.85):
            # if (tick['Local time'] <= 28000 and certainty >= 0.6):
                if (direction == 2):# and self.potential_stock < 1):
                    self.twos += 1
                    self.potential_stock += tick['AskVolume']
                    self.sell_orders.append((self.tick_num + self.pred_horizon, microprice))
                    new_orders.append({'type': 'BID', 'price': microprice, 'quantity': tick['AskVolume']})
                else:#if(direction == 0):#if (self.potential_stock > -1):
                    self.ones += 1
                    self.potential_stock -= tick['BidVolume']
                    self.buy_orders.append((self.tick_num + self.pred_horizon, microprice))
                    new_orders.append({'type': 'ASK', 'price': microprice, 'quantity': tick['BidVolume']})
        
        if len(new_orders) == 0:
            return None
        else:
            return new_orders

    def filled_order(self, order):
        if (order['type'] == 'BID'):
            self.stock += order['quantity']
            self.balance -= order['price'] * order['quantity']
        else:
            self.stock -= order['quantity']
            self.balance += order['price'] * order['quantity']
        # print(self.balance + (self.stock * self.micros[-1]))
        # print(self.stock, self.balance)

    def print_vals(self):
        print("ones: " + str(self.ones) + " twos: " + str(self.twos))
        print("net worth: " + str(self.balance + (self.stock * self.micros[-1])))
        print("balance: " + str(self.balance))
        print("quantity: " + str(self.stock))
        print("profit: " + str(self.balance - 4000))
        print("std_dev: " + str(statistics.stdev(self.diffs)))

        return ([self.balance - 4000, self.ones, self.twos, statistics.stdev(self.diffs)])

    def __gen_inputs__(self):
        inputs = []
        timespans = []
        for i, tick in enumerate(self.ticks):
            if (i < 50): 
                continue

            inputs.append([
                (tick['Bid'] - self.bid_norm_vals['min_p']) / (self.bid_norm_vals['max_p'] - self.bid_norm_vals['min_p']),
                (tick['Ask'] - self.ask_norm_vals['min_p']) / (self.ask_norm_vals['max_p'] - self.ask_norm_vals['min_p']),
                (tick['AskVolume'] - self.ask_norm_vals['min_v']) / (self.ask_norm_vals['max_v'] - self.ask_norm_vals['min_v']),
                (tick['BidVolume'] - self.bid_norm_vals['min_v']) / (self.bid_norm_vals['max_v'] - self.bid_norm_vals['min_v'])
            ])
            timespans.append(tick['Local time'])
        
        for i, mic in enumerate(self.micros):
            if (i < 50): 
                continue

            inputs[i-50].append(
                (mic - self.mic_norm_vals['min_p']) / (self.mic_norm_vals['max_p'] - self.mic_norm_vals['min_p'])
            )

        for i, e in enumerate(self.emas):
            if (i < 50): 
                continue

            inputs[i-50].append(
                (e - self.ema_norm_vals['min_p']) / (self.ema_norm_vals['max_p'] - self.ema_norm_vals['min_p'])
            )
            inputs[i-50].append(self.micros[-1] - self.micros[i])
        
        return (torch.Tensor(np.asarray([inputs])).float(), 
                torch.Tensor(np.asarray([timespans])).float())
    
    def __update_norm_vals__(self):
        self.bid_norm_vals['max_p'] = max(self.bid_norm_vals['max_p'], self.ticks[-1]['Bid'])
        self.bid_norm_vals['min_p'] = min(self.bid_norm_vals['min_p'], self.ticks[-1]['Bid'])
        self.bid_norm_vals['max_v'] = max(self.bid_norm_vals['max_v'], self.ticks[-1]['BidVolume'])
        self.bid_norm_vals['min_v'] = min(self.bid_norm_vals['min_v'], self.ticks[-1]['BidVolume'])

        self.ask_norm_vals['max_p'] = max(self.ask_norm_vals['max_p'], self.ticks[-1]['Ask'])
        self.ask_norm_vals['min_p'] = min(self.ask_norm_vals['min_p'], self.ticks[-1]['Ask'])
        self.ask_norm_vals['max_v'] = max(self.ask_norm_vals['max_v'], self.ticks[-1]['AskVolume'])
        self.ask_norm_vals['min_v'] = min(self.ask_norm_vals['min_v'], self.ticks[-1]['AskVolume'])

        self.mic_norm_vals['max_p'] = max(self.mic_norm_vals['max_p'], self.micros[-1])
        self.mic_norm_vals['min_p'] = min(self.mic_norm_vals['min_p'], self.micros[-1])

        self.ema_norm_vals['max_p'] = max(self.ema_norm_vals['max_p'], self.emas[-1])
        self.ema_norm_vals['min_p'] = min(self.ema_norm_vals['min_p'], self.emas[-1])


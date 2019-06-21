import torch
import torch.nn as nn


for name,para in nn.LSTM(3,3).named_parameters():
    print(name,para)
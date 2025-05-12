import torch
import torch.nn as nn
import numpy as np


class DeltaNetwork(nn.Module):
    def __init__(self, input_dim=2048, layers=2, init_val=2.0):
        super(DeltaNetwork, self).__init__()
        if layers == 2:
            self.delta = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
            )
        elif layers == 3:
            self.delta = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1)
            )
        elif layers == 5:
            self.delta = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 1)
            )

        for layer in self.delta:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.init_val = init_val
        nn.init.constant_(self.delta[-1].bias, init_val)  # Set bias to the calculated value
        # self.delta[-1].bias.requires_grad = False
    def forward(self, x):
        return self.delta(x)

class GammaNetwork(nn.Module):
    def __init__(self, input_dim=2048, layers=2, init_val=0.25):
        super(GammaNetwork, self).__init__()
        if layers == 2:
            self.gamma = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        elif layers == 3:
            self.gamma = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif layers == 5:
            self.gamma = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        for layer in self.gamma:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.init_val = init_val
        nn.init.constant_(self.gamma[-2].bias, np.log(init_val / (1 - init_val)))  # Set bias to the calculated value
        # self.gamma[-2].bias.requires_grad = False
    def forward(self, x):
        return self.gamma(x)

class GammaLSTM(nn.Module):
    # def __init__(self, input_dim=2048, hidden_size=256, init_val=0.25, device="cpu"):
    #     super(GammaLSTM, self).__init__()
    #     self.hidden_size = hidden_size
    #     self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
    #     self.fc = nn.Linear(hidden_size, 1)
    #     self.activation = nn.Sigmoid()
    #     self.init_val = init_val
    #     nn.init.constant_(self.fc.bias, np.log(init_val / (1 - init_val)))  # Set bias to the calculated value
    #     self.device = device
    #     self.to(self.device) 

    def __init__(self, input_dim=2048, hidden_size=256, init_val=0.25, device="cpu"):
        super(GammaLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()
        
        # Set bias to achieve the desired initial output for sigmoid
        self.init_val = init_val
        nn.init.constant_(self.fc.bias, torch.log(torch.tensor(init_val / (1 - init_val))))
        # self.fc.bias.requires_grad = False
        
        # Initialize weights for the LSTM and FC layer
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform_(self.fc.weight)  # Xavier for the FC layer
        
        self.device = device
        self.to(self.device)
        self.quantize = False

    def forward(self, x):
        # Initialize hidden and cell state
        if self.quantize:
            dtype = torch.float16
        else:
            dtype = torch.float
        if x.dim() == 3: # 3D
            h0 = torch.zeros(1, x.size(0), self.hidden_size, device=self.device, dtype=dtype)
            c0 = torch.zeros(1, x.size(0), self.hidden_size, device=self.device, dtype=dtype)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
        else: # 2D
            h0 = torch.zeros(1, self.hidden_size, device=self.device, dtype=dtype)
            c0 = torch.zeros(1, self.hidden_size, device=self.device, dtype=dtype)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[-1, :])
        
        
        out = self.activation(out) 
        return out

class DeltaLSTM(nn.Module):
    # def __init__(self, input_dim=2048, hidden_size=256, init_val=2.0, device="cpu"):
    #     super(DeltaLSTM, self).__init__()
    #     self.hidden_size = hidden_size
    #     self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
    #     self.fc = nn.Linear(hidden_size, 1)
    #     self.init_val = init_val
    #     nn.init.constant_(self.fc.bias, init_val)  # Set bias to the calculated value
    #     self.device = device
    #     self.to(self.device)

    def __init__(self, input_dim=2048, hidden_size=256, init_val=2.0, device="cpu"):
        super(DeltaLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.init_val = init_val
        
        # Initialize weights for LSTM
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.fc.bias, init_val)  # Set bias to the calculated value
        nn.init.xavier_uniform_(self.fc.weight)    # Xavier initialization for weights
        # self.fc.bias.requires_grad = False

        self.device = device
        self.to(self.device) 
        self.quantize = False

    def forward(self, x):
        # Initialize hidden and cell state
        if self.quantize:
            dtype = torch.float16
        else:
            dtype = torch.float
        if x.dim() == 3: # 3D, batch_size * len * emb_dim
            h0 = torch.zeros(1, x.size(0), self.hidden_size, device=self.device, dtype=dtype)
            c0 = torch.zeros(1, x.size(0), self.hidden_size, device=self.device, dtype=dtype)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
        else: # 2D, len * emb_dim
            h0 = torch.zeros(1, self.hidden_size, device=self.device, dtype=dtype)
            c0 = torch.zeros(1, self.hidden_size, device=self.device, dtype=dtype)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[-1, :])
        return out

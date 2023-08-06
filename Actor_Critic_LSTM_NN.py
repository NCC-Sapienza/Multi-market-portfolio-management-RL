import torch
import torch.nn as nn
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, space_dim, action_dim, HIDDEN_DIM=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(in_features=space_dim, out_features=HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_DIM, out_features=HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_DIM, out_features=action_dim),
            nn.Sigmoid()
        )

    def forward(self,x):
        dist = self.actor(x)
        return dist

    def save_dict(self):
        torch.save(self.state_dict(), "tmp/actor_model.pth")
    def load_dict(self):
        self.load_state_dict(torch.load("tmp/actor_model.pth"))

class CriticNetwork(nn.Module):
    def __init__(self,space_dim, HIDDEN_DIM=256, critic_value=1):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(in_features=space_dim, out_features=HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_DIM, out_features=HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_DIM, out_features=critic_value),
        )

    def forward(self, x):
        return self.critic(x)

    def save_dict(self):
        torch.save(self.state_dict(), "tmp/critic_model.pth")
    def load_dict(self):
        self.load_state_dict(torch.load("tmp/critic_model.pth"))


class LSTM_model(nn.Module):
    def __init__(self, input_dim, hidden, num_layer, output_dim):
        super().__init__()
        self.dim = input_dim
        self.hidden = hidden
        self.output = output_dim
        self.emb_layer = nn.Sequential(
            nn.Linear(self.dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU())
        self.out_layer = nn.Linear(hidden, output_dim)
        self.rnn_layer = num_layer
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, num_layers=self.rnn_layer, batch_first=True)

    def forward(self, input_data):
        x = self.emb_layer(input_data)
        output, hidden_ = self.lstm(x)
        out = self.out_layer(output[:, -1, :]).squeeze()
        return out



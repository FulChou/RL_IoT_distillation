# 作者：vincent
# code time：2021/12/20 22:11
from torch import nn
import torch
import numpy as np

class TeacherNet(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs,device=self.device, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

class TeacherNet_lunar(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs,device=self.device, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class StudentNet(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 26), nn.ReLU(inplace=True),
            nn.Linear(26, 2), nn.ReLU(inplace=True),
            nn.Linear(2, 17), nn.ReLU(inplace=True),
            nn.Linear(17, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs,device=self.device, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class StudentNet_lunar(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 4), nn.ReLU(inplace=True),
            nn.Linear(4, 19), nn.ReLU(inplace=True),
            nn.Linear(19, 4), nn.ReLU(inplace=True),
            nn.Linear(4, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs,device=self.device, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, input_shape: tuple, n_actions: int):
        super().__init__()

        # Calculate flattened input size
        self.n_nodes = input_shape[0]
        self.n_edges = input_shape[1]

        # Network architecture
        self.fc1 = nn.Linear(self.n_nodes + self.n_edges + 5, 128)  # +5 for slice info
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        """Handle both batch and single state inputs"""
        if isinstance(x, tuple):  # Batch from replay buffer
            # Convert tuple of dicts to dict of tensors
            cpu_usage = torch.FloatTensor([s["cpu_usage"] for s in x])
            bw_usage = torch.FloatTensor([s["bandwidth_usage"] for s in x])
            slice_info = torch.FloatTensor(
                [
                    [
                        s["current_slice"]["slice_type"],
                        *s["current_slice"]["qos"],
                        s["current_slice"]["vnfs_placed"],
                    ]
                    for s in x
                ]
            )
        else:  # Single state
            cpu_usage = torch.FloatTensor(x["cpu_usage"])
            bw_usage = torch.FloatTensor(x["bandwidth_usage"])
            slice_info = torch.FloatTensor(
                [
                    x["current_slice"]["slice_type"],
                    *x["current_slice"]["qos"],
                    x["current_slice"]["vnfs_placed"],
                ]
            )

        x = torch.cat([cpu_usage, bw_usage, slice_info], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self, state: dict, action: int, reward: float, next_state: dict, done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

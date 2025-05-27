# models/dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, input_shape: tuple, n_actions: int):
        super().__init__()
        self.n_nodes = input_shape[0]  # From observation_space['cpu_usage']
        self.n_edges = input_shape[1]  # From observation_space['bandwidth_usage']

        # Correct input size calculation:
        # cpu_usage: n_nodes
        # bandwidth_usage: n_edges
        # slice_info: type (1) + qos (3) + vnfs_placed (1) = 5
        total_input_size = self.n_nodes + self.n_edges + 5

        self.fc1 = nn.Linear(total_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Handles both single states and batches exactly as the environment provides them"""
        if isinstance(x, dict):  # Single state
            # Convert each component to proper tensor format
            cpu = torch.FloatTensor(x["cpu_usage"]).unsqueeze(0)  # [1, n_nodes]
            bw = torch.FloatTensor(x["bandwidth_usage"]).unsqueeze(0)  # [1, n_edges]
            slice_info = torch.FloatTensor(
                [
                    float(x["current_slice"]["slice_type"]),  # 1
                    *x["current_slice"]["qos"],  # 3
                    float(x["current_slice"]["vnfs_placed"]),  # 1
                ]
            ).unsqueeze(0)  # [1, 5]
        else:  # Batch from replay buffer
            # Process each component for all states in batch
            cpu = torch.stack(
                [torch.FloatTensor(s["cpu_usage"]) for s in x]
            )  # [batch, n_nodes]
            bw = torch.stack(
                [torch.FloatTensor(s["bandwidth_usage"]) for s in x]
            )  # [batch, n_edges]
            slice_info = torch.stack(
                [
                    torch.FloatTensor(
                        [
                            float(s["current_slice"]["slice_type"]),
                            *s["current_slice"]["qos"],
                            float(s["current_slice"]["vnfs_placed"]),
                        ]
                    )
                    for s in x
                ]
            )  # [batch, 5]

        # Concatenate all features along the last dimension
        x = torch.cat([cpu, bw, slice_info], dim=-1)  # [batch, n_nodes + n_edges + 5]
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

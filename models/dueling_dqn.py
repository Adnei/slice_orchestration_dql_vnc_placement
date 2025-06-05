import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, input_shape: tuple, n_actions: int):
        super().__init__()
        self.n_nodes = input_shape[0]  # From observation_space['cpu_usage']
        self.n_edges = input_shape[1]  # From observation_space['bandwidth_usage']

        # Input features: cpu_usage (n_nodes), bandwidth_usage (n_edges), slice_type (1), qos (3), vnfs_placed (1)
        self.total_input_size = self.n_nodes + self.n_edges + 5

        self.feature = nn.Sequential(
            nn.Linear(self.total_input_size, 128),
            nn.ReLU(),
        )

        # Value stream (estimates state value V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Advantage stream (estimates A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        if isinstance(x, dict):  # Single state
            cpu = torch.FloatTensor(x["cpu_usage"]).unsqueeze(0)  # [1, n_nodes]
            bw = torch.FloatTensor(x["bandwidth_usage"]).unsqueeze(0)  # [1, n_edges]
            slice_info = torch.FloatTensor(
                [
                    float(x["current_slice"]["slice_type"]),
                    *x["current_slice"]["qos"],
                    float(x["current_slice"]["vnfs_placed"]),
                ]
            ).unsqueeze(0)  # [1, 5]
        else:  # Batch
            cpu = torch.stack([torch.FloatTensor(s["cpu_usage"]) for s in x])
            bw = torch.stack([torch.FloatTensor(s["bandwidth_usage"]) for s in x])
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
            )

        flat_state = torch.cat(
            [cpu, bw, slice_info], dim=-1
        )  # [batch, total_input_size]
        features = self.feature(flat_state)
        value = self.value_stream(features)  # [batch, 1]
        advantage = self.advantage_stream(features)  # [batch, n_actions]

        # Combine streams: Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

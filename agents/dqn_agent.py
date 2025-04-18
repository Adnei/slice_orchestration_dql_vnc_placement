import torch
import numpy as np
from typing import Dict
from collections import deque
import random

from models.dqn import DQN, ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_shape: tuple,
        n_actions: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
    ):
        self.invalid_penalty = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0
        self.n_actions = n_actions

    def select_action(self, state: Dict, valid_nodes: list) -> int:
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.choice(valid_nodes)

        with torch.no_grad():
            # Get Q-values for all nodes
            q_values = self.policy_net(state).cpu().numpy()

            # Filter Q-values for valid nodes only
            valid_q_values = [q_values[node] for node in valid_nodes]

            # Select node with highest Q-value among valid nodes
            return valid_nodes[np.argmax(valid_q_values)]

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors - now handles dict states properly
        state_batch = states  # Pass directly to network
        action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float)
        next_state_batch = next_states  # Pass directly
        done_batch = torch.tensor(dones, dtype=torch.float)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1})
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]

        # Compute expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma * (1 - done_batch)
        ) + reward_batch

        # Compute loss
        loss = torch.nn.functional.mse_loss(
            state_action_values.squeeze(), expected_state_action_values
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]

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
        lr: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.992,
        buffer_size: int = 10000,
        batch_size: int = 128,
        target_update: int = 100,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions

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
        self.last_100_rewards = deque(maxlen=100)

    def select_action(self, state: Dict, valid_nodes: list) -> int:
        if not valid_nodes:
            return -1  # Indicate no valid nodes

        self.steps_done += 1

        # Adaptive epsilon based on recent performance
        if len(self.last_100_rewards) == 100 and np.mean(self.last_100_rewards) < -1.0:
            self.epsilon = min(
                0.5, self.epsilon * 1.2
            )  # Boost exploration if struggling

        if random.random() < self.epsilon:
            return random.choice(valid_nodes)

        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy()[0]
            valid_q_values = {node: q_values[node] for node in valid_nodes}
            return max(valid_q_values.items(), key=lambda x: x[1])[0]

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)

        # Current Q values
        state_action_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Expected Q values
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            expected_state_action_values = (
                rewards + (1 - dones) * self.gamma * next_state_values
            )

        # Compute loss
        loss = torch.nn.functional.mse_loss(
            state_action_values, expected_state_action_values
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 1.0
        )  # Gradient clipping
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def update_reward_history(self, reward):
        self.last_100_rewards.append(reward)

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

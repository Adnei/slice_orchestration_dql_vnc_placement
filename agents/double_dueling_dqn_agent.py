import torch
import numpy as np
from typing import Dict
from collections import deque
import random
from models.dueling_dqn import DuelingDQN
from models.replay_buffer import ReplayBuffer


class D3QNAgent:
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
        eval_mode: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.eval_mode = eval_mode

        self.policy_net = DuelingDQN(state_shape, n_actions).to(self.device)
        self.target_net = DuelingDQN(state_shape, n_actions).to(self.device)
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
            return -1  # Special value indicating no valid nodes

        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy()[0]
            valid_q_values = {node: q_values[node] for node in valid_nodes}

        if self.eval_mode:
            return max(valid_q_values.items(), key=lambda x: x[1])[0]

        explore_prob = self.epsilon
        if len(valid_nodes) < 3:
            explore_prob = max(self.epsilon, 0.5)

        if random.random() < explore_prob:
            return random.choice(valid_nodes)
        else:
            return max(valid_q_values.items(), key=lambda x: x[1])[0]

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)

        state_action_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.policy_net(next_states)
            next_actions = next_q_values.argmax(1, keepdim=True)
            next_q_target_values = (
                self.target_net(next_states).gather(1, next_actions).squeeze(1)
            )
            expected_state_action_values = (
                rewards + (1 - dones) * self.gamma * next_q_target_values
            )

        loss = torch.nn.functional.mse_loss(
            state_action_values, expected_state_action_values
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1
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

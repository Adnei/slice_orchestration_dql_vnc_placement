from simple_vnf_env import SimpleVNFEnv, VNF, VNFType
from simple_dqn_agent import SimpleDQNAgent
import numpy as np
import random


def create_simple_slice(episode):
    # Start with simple slices, increase complexity over time
    max_vnfs = 2 if episode < 200 else 3 if episode < 500 else 4
    vnfs = [
        VNF(0, VNFType.RAN, cpu_demand=5, bandwidth_demand=10),
        VNF(1, VNFType.EDGE, cpu_demand=10, bandwidth_demand=20),
        VNF(2, VNFType.TRANSPORT, cpu_demand=15, bandwidth_demand=30),
        VNF(3, VNFType.CORE, cpu_demand=20, bandwidth_demand=40),
    ]
    return vnfs[: random.randint(2, max_vnfs)]


def train():
    env = SimpleVNFEnv(n_nodes=20)
    agent = SimpleDQNAgent(state_size=20 * 4, action_size=20)  # 20 nodes * 4 features

    episodes = 1000
    for ep in range(episodes):
        state = env.reset()
        slice_vnfs = create_simple_slice(ep)
        env.add_slice(slice_vnfs)

        total_reward = 0
        done = False

        while not done:
            # Get valid actions (nodes that can host current VNF)
            current_vnf = slice_vnfs[env.current_vnf_idx]
            valid_actions = [
                node
                for node in env.topology.nodes()
                if (
                    env.topology.nodes[node]["type"] == current_vnf.vnf_type
                    and env.topology.nodes[node]["cpu_used"] + current_vnf.cpu_demand
                    <= env.topology.nodes[node]["cpu_capacity"]
                )
            ]

            if not valid_actions:
                # Penalize for not being able to complete slice
                reward = -2 * len(slice_vnfs)  # Scale penalty with slice size
                total_reward += reward
                break

            action = agent.select_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")


if __name__ == "__main__":
    train()

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List
from environment.vnf_env import VNFPlacementEnv
from agents.dqn_agent import DQNAgent
from environment.network_slice import NetworkSlice, QoS, SliceType, VNF
from environment.barabasi_network_generator import NetworkTopologyGenerator

# from environment.network_topology_generator import NetworkTopologyGenerator


class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.energy_consumptions = []
        self.invalid_actions = []
        self.qos_violations = []

    def update(self, episode_reward, energy, invalid_count, qos_violated):
        self.episode_rewards.append(episode_reward)
        self.energy_consumptions.append(energy)
        self.invalid_actions.append(invalid_count)
        self.qos_violations.append(qos_violated)

    def plot(self):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")

        plt.subplot(2, 2, 2)
        plt.plot(self.energy_consumptions)
        plt.title("Energy Consumption")
        plt.xlabel("Episode")
        plt.ylabel("Total Energy")

        plt.subplot(2, 2, 3)
        plt.plot(self.invalid_actions)
        plt.title("Invalid Actions")
        plt.xlabel("Episode")
        plt.ylabel("Count")

        plt.subplot(2, 2, 4)
        plt.plot(self.qos_violations)
        plt.title("QoS Violations")
        plt.xlabel("Episode")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig("training_metrics.png")
        plt.show()


def create_sample_slices(topology: nx.Graph, n_slices: int = 2) -> List[NetworkSlice]:
    slices = []
    ran_nodes = [n for n in topology.nodes() if topology.nodes[n]["type"] == "RAN"]

    for i in range(n_slices):
        slice_type = random.choice(list(SliceType))

        if slice_type == SliceType.URLLC:
            qos = QoS(qos_id=i, max_latency=100, edge_latency=50, min_bandwidth=5)
            vnf_cpu = random.randint(1, 2)
        elif slice_type == SliceType.EMBB:
            qos = QoS(qos_id=i, max_latency=200, min_bandwidth=10)
            vnf_cpu = random.randint(1, 2)
        else:
            qos = QoS(qos_id=i, max_latency=300, min_bandwidth=2)
            vnf_cpu = random.randint(1, 2)

        slice = NetworkSlice(
            slice_id=i, slice_type=slice_type, qos=qos, origin=random.choice(ran_nodes)
        )

        for vnf_type in ["RAN", "Edge", "Transport", "Core"]:
            slice.add_vnf(
                VNF(
                    vnf_id=len(slice.vnf_list),
                    delay=random.uniform(0.1, 0.5),
                    vnf_type=vnf_type,
                    vcpu_usage=vnf_cpu,
                    bandwidth_usage=max(
                        1, qos.min_bandwidth * random.uniform(0.1, 0.3)
                    ),
                )
            )

        slices.append(slice)

    return slices


def train_dqn_agent():
    # Initialize metrics tracker
    metrics = TrainingMetrics()

    # Generate network topology
    topology_generator = NetworkTopologyGenerator(n_nodes=100, avg_degree=4)
    topology = topology_generator.get_graph()
    topology_generator.draw()

    # Create environment
    env = VNFPlacementEnv(topology)

    # Create DQN agent
    state_shape = (len(topology.nodes()), len(topology.edges()))
    n_actions = len(topology.nodes())
    agent = DQNAgent(
        state_shape,
        n_actions,
        lr=0.0005,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=0.992,
        batch_size=128,
        target_update=100,
    )

    # Training parameters
    n_episodes = 5000
    batch_size = 64
    print_interval = 50

    for episode in range(n_episodes):
        # Generate new slices for this episode

        # Disabling URLLC for now
        slices = create_sample_slices(topology, n_slices=2)

        # Gradually increase difficulty
        if episode > 200 and episode % 100 == 0:
            n_slices = min(5, 2 + episode // 200)
            slices = create_sample_slices(topology, n_slices=n_slices)

        episode_reward = 0
        episode_energy = 0
        invalid_action_count = 0
        qos_violated = 0

        for slice in slices:
            # Reset environment for new slice
            state, _ = env.reset()
            env.add_slice(slice)

            terminated = False
            while not terminated:
                # Get valid nodes for current VNF
                current_vnf_idx = len(slice.path or [])
                if current_vnf_idx >= len(slice.vnf_list):
                    break

                current_vnf = slice.vnf_list[current_vnf_idx]
                valid_nodes = [
                    node
                    for node in topology.nodes()
                    if (
                        topology.nodes[node]["type"] == current_vnf.vnf_type
                        and (topology.nodes[node]["cpu_usage"] + current_vnf.vcpu_usage)
                        <= topology.nodes[node]["cpu_limit"]
                    )
                ]

                if not valid_nodes:
                    invalid_action_count += 1
                    break

                # Select action
                action = agent.select_action(state, valid_nodes)

                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                # Track invalid actions from rewards
                if reward == -env.max_energy_per_step:
                    invalid_action_count += 1

                # Store transition in memory
                agent.memory.push(state, action, reward, next_state, terminated)

                # Move to next state
                state = next_state

                # Optimize model
                loss = agent.optimize_model()

            # Check QoS after placement
            if slice.path and not slice.validate_vnf_placement(topology):
                qos_violated += 1

            # Calculate energy for this slice
            episode_energy += sum(
                topology.nodes[node]["energy_base"]
                + topology.nodes[node]["energy_per_vcpu"]
                * topology.nodes[node]["cpu_usage"]
                for node in slice.path or []
            )

        # Update metrics
        metrics.update(
            episode_reward, episode_energy, invalid_action_count, qos_violated
        )

        # Evaluation and logging
        if episode % print_interval == 0:
            avg_reward = np.mean(metrics.episode_rewards[-print_interval:])
            avg_energy = np.mean(metrics.energy_consumptions[-print_interval:])
            print(
                f"Episode {episode} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Energy: {avg_energy:.2f} | "
                f"Invalid Actions: {invalid_action_count} | "
                f"QoS Violations: {qos_violated}"
            )

        agent.update_reward_history(episode_reward)

    # Save trained agent and metrics
    agent.save("dqn_agent.pth")
    metrics.plot()


if __name__ == "__main__":
    train_dqn_agent()

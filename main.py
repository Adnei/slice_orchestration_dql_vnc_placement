import numpy as np
import torch
import networkx as nx
import random
from typing import List
import matplotlib.pyplot as plt
from collections import deque

from environment.vnf_env import VNFPlacementEnv
from agents.dqn_agent import DQNAgent
from environment.network_slice import NetworkSlice, QoS, SliceType, VNF
from utils.evaluator import Evaluator
from network_topology_generator import NetworkTopologyGenerator


def create_sample_slices(topology: nx.Graph, n_slices: int = 10) -> List[NetworkSlice]:
    """Generate sample network slices for training/testing"""
    slices = []
    ran_nodes = [
        node for node in topology.nodes() if topology.nodes[node]["type"] == "RAN"
    ]

    for i in range(n_slices):
        # Randomly select slice type
        slice_type = random.choice(list(SliceType))

        # Create appropriate QoS
        if slice_type == SliceType.URLLC:
            qos = QoS(
                qos_id=i,
                max_latency=1.0,  # 1ms for URLLC
                edge_latency=0.5,  # Strict edge processing
                min_bandwidth=100,  # Mbps
            )
        elif slice_type == SliceType.EMBB:
            qos = QoS(
                qos_id=i,
                max_latency=10.0,
                min_bandwidth=1000,  # 10ms for eMBB  # 1Gbps
            )
        else:  # mMTC or GENERIC
            qos = QoS(qos_id=i, max_latency=100.0, min_bandwidth=10)  # 100ms  # 10Mbps

        # Select random RAN node as origin
        origin = random.choice(ran_nodes)

        # Create slice
        network_slice = NetworkSlice(
            slice_id=i, slice_type=slice_type, qos=qos, origin=origin
        )

        # Add VNFs (simplified - in reality this would be slice-specific)
        vnf_types = ["RAN", "Edge", "Transport", "Core"]
        for vnf_type in vnf_types:
            network_slice.add_vnf(
                VNF(
                    vnf_id=len(network_slice.vnf_list),
                    delay=random.uniform(0.1, 0.5),
                    vnf_type=vnf_type,
                    vcpu_usage=random.randint(1, 4),
                    bandwidth_usage=random.uniform(10, 100),
                )
            )

        slices.append(network_slice)

    return slices


def train_dqn_agent():
    # Generate network topology
    topology_generator = NetworkTopologyGenerator(n_nodes=50, avg_degree=4)
    topology = topology_generator.get_graph()

    # Create environment
    env = VNFPlacementEnv(topology)

    # Create DQN agent
    state_shape = (len(topology.nodes()), len(topology.edges()))
    n_actions = len(topology.nodes())
    agent = DQNAgent(state_shape, n_actions)

    # Training parameters
    n_episodes = 1000
    batch_size = 64
    print_interval = 50

    # Training loop
    rewards_history = []
    energy_history = []

    for episode in range(n_episodes):
        # Generate new slices for this episode
        slices = create_sample_slices(topology, n_slices=5)

        total_reward = 0
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
                    break

                # Select action
                action = agent.select_action(state, valid_nodes)

                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                # Store transition in memory
                agent.memory.push(state, action, reward, next_state, terminated)

                # Move to next state
                state = next_state

                # Optimize model
                loss = agent.optimize_model()

        # Log progress
        rewards_history.append(total_reward)

        # Evaluate periodically
        if episode % print_interval == 0:
            evaluator = Evaluator(topology)
            test_slices = create_sample_slices(topology, n_slices=10)
            energy = evaluator.evaluate_model(agent, test_slices)
            energy_history.append(energy)

            print(
                f"Episode {episode}, Reward: {total_reward:.2f}, Energy: {energy:.2f}"
            )

    # Save trained agent
    agent.save("dqn_agent.pth")

    # Plot training results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.plot(energy_history)
    plt.title("Energy Consumption")
    plt.xlabel("Evaluation Interval")
    plt.ylabel("Total Energy")

    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()


if __name__ == "__main__":
    train_dqn_agent()

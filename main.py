import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List
from environment.vnf_env import VNFPlacementEnv
from slice_scheduler import SliceScheduler

# from agents.dqn_agent import DQNAgent
from agents.double_dueling_dqn_agent import D3QNAgent as DQNAgent
from environment.network_slice import NetworkSlice, QoS, SliceType, VNF
import sys

# from environment.network_topology_generator import NetworkTopologyGenerator
from environment.barabasi_network_generator import NetworkTopologyGenerator
from topology_visualizer import TopologyVisualizer


class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.energy_consumptions = []
        self.invalid_actions = []
        self.qos_violations = []
        self.successful_placements = []
        self.qos_success_rate = []
        self.num_slices_per_episode = []

    def update(
        self,
        episode_reward,
        energy,
        invalid_count,
        qos_violated,
        success_count,
        num_slices,
    ):
        self.episode_rewards.append(episode_reward)
        self.energy_consumptions.append(energy)
        self.invalid_actions.append(invalid_count)
        self.qos_violations.append(qos_violated)
        self.successful_placements.append(success_count)
        self.qos_success_rate.append(
            (success_count - qos_violated) / max(success_count, 1)
        )
        self.num_slices_per_episode.append(num_slices)

    def plot(self):
        plt.figure(figsize=(15, 10))

        # Smooth rewards with moving average
        window_size = 50
        smoothed_rewards = np.convolve(
            self.episode_rewards, np.ones(window_size) / window_size, mode="valid"
        )

        x = list(range(len(self.episode_rewards)))
        slices = self.num_slices_per_episode

        def plot_with_slices(x, y, title, ylabel, smoothed=None, subplot_idx=1):
            ax1 = plt.subplot(2, 2, subplot_idx)
            ax1.plot(x, y, alpha=0.3, label="Raw")
            if smoothed is not None:
                ax1.plot(
                    x[window_size - 1 :], smoothed, label="Smoothed", color="orange"
                )
            ax1.set_title(title)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel(ylabel)
            ax1.legend(loc="upper left")

            ax2 = ax1.twinx()
            ax2.plot(
                x, slices, label="# Slices", color="gray", linestyle="--", alpha=0.6
            )
            ax2.set_ylabel("Num Slices", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

        plot_with_slices(
            x,
            self.episode_rewards,
            "Episode Rewards",
            "Total Reward",
            smoothed_rewards,
            1,
        )
        plot_with_slices(
            x, self.energy_consumptions, "Energy Consumption", "Total Energy", None, 2
        )
        plot_with_slices(x, self.invalid_actions, "Invalid Actions", "Count", None, 3)
        plot_with_slices(x, self.qos_violations, "QoS Violations", "Count", None, 4)

        plt.tight_layout()
        plt.savefig("training_metrics.png")
        plt.close()

    def plot_convergence_curve(self, filename="convergence_curve.png"):
        plt.figure(figsize=(12, 6))

        def smooth(values, alpha=0.1):
            ema = [values[0]]
            for v in values[1:]:
                ema.append(alpha * v + (1 - alpha) * ema[-1])
            return ema

        episodes = np.arange(len(self.episode_rewards))
        smoothed_rewards = smooth(self.episode_rewards)
        smoothed_energy = smooth(self.energy_consumptions)
        smoothed_success = smooth(self.qos_success_rate)

        plt.plot(episodes, smoothed_rewards, label="Reward (EMA)", color="blue")
        plt.plot(episodes, smoothed_energy, label="Energy (EMA)", color="green")
        plt.plot(episodes, smoothed_success, label="Success Rate (EMA)", color="orange")

        plt.title("Convergence: Smoothed Reward, Energy, and Success Rate")
        plt.xlabel("Episode")
        plt.ylabel("Metric Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def create_sample_slices(
    topology: nx.Graph,
    n_slices: int = 2,
    valid_slice_types: list[SliceType] = list(SliceType),
) -> List[NetworkSlice]:
    slices = []
    ran_nodes = [
        node for node in topology.nodes() if topology.nodes[node]["type"] == "RAN"
    ]

    for i in range(n_slices):
        slice_type = random.choice(valid_slice_types)

        if slice_type == SliceType.URLLC:
            qos = QoS(qos_id=i, max_latency=4, edge_latency=0.8, min_bandwidth=100)
            vnf_cpu = random.randint(1, 2)
        # Enhanced Mobile Broadband
        elif slice_type == SliceType.EMBB:
            qos = QoS(qos_id=i, max_latency=10, min_bandwidth=1000)
            vnf_cpu = random.randint(1, 3)
        else:  # mMTC
            qos = QoS(qos_id=i, max_latency=100, min_bandwidth=50)
            vnf_cpu = random.randint(1, 2)

        origin = random.choice(ran_nodes)
        network_slice = NetworkSlice(
            slice_id=i, slice_type=slice_type, qos=qos, origin=origin
        )

        # Create VNF chain with type constraints
        for vnf_type in ["RAN", "Edge", "Transport", "Core"]:
            network_slice.add_vnf(
                VNF(
                    vnf_id=len(network_slice.vnf_list),
                    delay=random.uniform(0.1, 0.3),
                    vnf_type=vnf_type,
                    vcpu_usage=vnf_cpu,
                    bandwidth_usage=max(
                        1, qos.min_bandwidth * random.uniform(0.1, 0.3)
                    ),
                )
            )

        slices.append(network_slice)

    return slices


def train_dqn_agent(agent_load=None):
    metrics = TrainingMetrics()

    # Initialize network topology
    topology_generator = NetworkTopologyGenerator(n_nodes=50)
    topology_generator.draw()
    topology_generator.export_graph_to_pickle(filename="topology.pickle")
    topology = topology_generator.get_graph()

    # Create environment
    env = VNFPlacementEnv(topology)

    # Create DQN agent with optimized parameters
    state_shape = (len(topology.nodes()), len(topology.edges()))
    n_actions = len(topology.nodes())
    agent = DQNAgent(
        state_shape,
        n_actions,
        lr=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,  # Updated from 0.05 to 0.15 --> Trying to explore more (0.5)
        epsilon_decay=0.9995,
        buffer_size=20000,
        batch_size=128,
        target_update=200,
    )

    if agent_load:
        agent.load(agent_load)
    n_episodes = 5000
    print_interval = 50
    scheduler = SliceScheduler(strategy="log", min_slices=2, max_slices=124)

    for episode in range(n_episodes):
        n_slices = scheduler.get_num_slices(episode)
        slices = create_sample_slices(topology, n_slices=n_slices)

        episode_reward = 0
        episode_energy = 0
        invalid_action_count = 0
        qos_violated = 0
        successful_placements = 0
        state, _ = env.reset()
        for slice in slices:
            # Get initial valid nodes with path continuity check
            current_vnf_idx = 0
            # @TODO: Valid nodes should filter only neighbors
            valid_nodes = [
                node
                for node in topology.nodes()
                if (
                    topology.nodes[node]["type"]
                    == slice.vnf_list[current_vnf_idx].vnf_type
                    and (
                        topology.nodes[node]["cpu_usage"]
                        + slice.vnf_list[current_vnf_idx].vcpu_usage
                    )
                    <= topology.nodes[node]["cpu_limit"]
                )
            ]

            if not valid_nodes:
                invalid_action_count += 1
                continue

            env.add_slice(slice)

            terminated = False
            while not terminated:
                action = agent.select_action(state, valid_nodes)
                next_state, reward, terminated, _, _ = env.step(action)

                if action == -1:  # No valid nodes
                    invalid_action_count += 1
                    terminated = True
                    break

                episode_reward += reward

                if reward < 0 or not slice.validate_vnf_placement(topology):
                    qos_violated += 1

                # Store experience
                agent.memory.push(state, action, reward, next_state, terminated)

                # Optimize model
                loss = agent.optimize_model()

                # Move to next state
                state = next_state

                # Update valid nodes for next VNF if not terminated
                if not terminated:
                    current_vnf_idx = len(slice.path)
                    valid_nodes = [
                        node
                        for node in topology.nodes()
                        if (
                            topology.nodes[node]["type"]
                            == slice.vnf_list[current_vnf_idx].vnf_type
                            and (
                                topology.nodes[node]["cpu_usage"]
                                + slice.vnf_list[current_vnf_idx].vcpu_usage
                            )
                            <= topology.nodes[node]["cpu_limit"]
                            and (
                                not slice.path
                                or topology.has_edge(slice.path[-1], node)
                            )
                        )
                    ]

            if slice.path and len(slice.path) == len(slice.vnf_list):
                if slice.validate_vnf_placement(topology):
                    successful_placements += 1
                else:
                    qos_violated += 1
            else:
                qos_violated += 1

            episode_energy = env.total_energy_used(topology)

        metrics.update(
            episode_reward,
            episode_energy,
            invalid_action_count,
            qos_violated,
            successful_placements,
            n_slices,
        )
        # print(
        #     f"[DEBUG] Episode {episode}: Total Slices={n_slices} | "
        #     f"Success={successful_placements} | QoS Violated={qos_violated} | Invalid={invalid_action_count}"
        # )

        agent.update_reward_history(episode_reward)

        # Adaptive exploration adjustment (if things are too bad then we go back to exploration hehe)
        if episode > 100 and successful_placements / n_slices < 0.5:
            agent.epsilon = min(0.7, agent.epsilon * 1.1)

        if episode % print_interval == 0:
            avg_reward = np.mean(metrics.episode_rewards[-print_interval:])
            avg_energy = np.mean(metrics.energy_consumptions[-print_interval:])
            total_success = sum(metrics.successful_placements[-print_interval:])
            total_slices = sum(metrics.num_slices_per_episode[-print_interval:])
            success_rate = total_success / max(total_slices, 1)

            print(
                f"Episode {episode:5d} | "
                f"Slices: {n_slices:3d} | "
                f"Avg Reward: {avg_reward:10.2f} | "
                f"Avg Energy: {avg_energy:9.2f} | "
                f"Invalid: {invalid_action_count:3d} | "
                f"QoS Violations: {qos_violated:2d} | "
                f"Success Rate: {success_rate:5.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    # Save results
    # agent.save("heavy_training_15k_500slc.pth")
    agent.save("dqn_agent.pth")
    # agent.save("15k_episodes_dqn_agent.pth")
    metrics.plot()
    metrics.plot_convergence_curve()
    print("Training completed. Model saved to dqn_agent.pth")


if __name__ == "__main__":
    agent_load = sys.argv[1] if len(sys.argv) > 1 else None
    train_dqn_agent(agent_load=agent_load)

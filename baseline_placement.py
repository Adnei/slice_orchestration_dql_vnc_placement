import matplotlib.pyplot as plt
import networkx as nx
from environment.vnf_env import VNFPlacementEnv
from environment.network_slice import NetworkSlice
from environment.barabasi_network_generator import NetworkTopologyGenerator
from main import create_sample_slices
from agents.dqn_agent import DQNAgent
import copy
import numpy as np
from topology_visualizer import TopologyVisualizer
import plotly.graph_objects as go
import plotly.io as pio


def greedy_placement(env: VNFPlacementEnv, slice_obj: NetworkSlice) -> bool:
    env.add_slice(slice_obj)
    topology = env.topology

    for vnf in slice_obj.vnf_list:
        valid_nodes = [
            node
            for node in topology.nodes()
            if topology.nodes[node]["type"] == vnf.vnf_type
            and (topology.nodes[node]["cpu_usage"] + vnf.vcpu_usage)
            <= topology.nodes[node]["cpu_limit"]
            and (not slice_obj.path or topology.has_edge(slice_obj.path[-1], node))
        ]

        if not valid_nodes:
            return False

        chosen_node = valid_nodes[-1]
        _, reward, terminated, _, _ = env.step(chosen_node)
        if reward < 0 or (
            terminated and len(slice_obj.path) != len(slice_obj.vnf_list)
        ):
            return False

    return True


def random_valid_placement(env: VNFPlacementEnv, slice_obj: NetworkSlice) -> bool:
    import random

    env.add_slice(slice_obj)
    topology = env.topology

    for vnf in slice_obj.vnf_list:
        valid_nodes = [
            node
            for node in topology.nodes()
            if topology.nodes[node]["type"] == vnf.vnf_type
            and (topology.nodes[node]["cpu_usage"] + vnf.vcpu_usage)
            <= topology.nodes[node]["cpu_limit"]
            and (not slice_obj.path or topology.has_edge(slice_obj.path[-1], node))
        ]

        if not valid_nodes:
            return False

        chosen_node = random.choice(valid_nodes)
        _, reward, terminated, _, _ = env.step(chosen_node)
        if reward < 0 or (
            terminated and len(slice_obj.path) != len(slice_obj.vnf_list)
        ):
            return False

    return True


def dqn_agent_placement(
    env: VNFPlacementEnv, slice_obj: NetworkSlice, agent: DQNAgent
) -> bool:
    env.add_slice(slice_obj)
    topology = env.topology
    state = env.get_observation()

    for vnf in slice_obj.vnf_list:
        valid_nodes = [
            node
            for node in topology.nodes()
            if topology.nodes[node]["type"] == vnf.vnf_type
            and (topology.nodes[node]["cpu_usage"] + vnf.vcpu_usage)
            <= topology.nodes[node]["cpu_limit"]
            and (not slice_obj.path or topology.has_edge(slice_obj.path[-1], node))
        ]

        if not valid_nodes:
            return False

        action = agent.select_action(state, valid_nodes)
        next_state, reward, terminated, _, _ = env.step(action)
        state = next_state

        if reward < 0 or (
            terminated and len(slice_obj.path) != len(slice_obj.vnf_list)
        ):
            return False

    return True


def evaluate_energy_vs_slices(topology, agent, max_slices=400, show_topology=True):
    strategies = {
        "Greedy": greedy_placement,
        "Random": random_valid_placement,
        "DQN": lambda env, s: dqn_agent_placement(env, s, agent),
    }

    energy_data = {k: [] for k in strategies}
    invalid_data = {k: [] for k in strategies}

    for n in range(1, max_slices + 1):
        slices = create_sample_slices(topology, n_slices=n)

        for name, strategy in strategies.items():
            env = VNFPlacementEnv(topology)
            env.reset()
            success = 0
            invalid = 0
            new_slices = copy.deepcopy(slices)
            for s in new_slices:
                if strategy(env, s):
                    success += 1
                else:
                    invalid += 1

            energy = env.total_energy_used(env.topology)
            energy_data[name].append(energy)
            invalid_data[name].append(invalid)
            if show_topology and n == range(1, max_slices + 1)[-1]:
                visualizer = TopologyVisualizer(topology)
                visualizer.animate_slice_building(
                    new_slices, complete_fig_name=f"{name}_placement"
                )

    return energy_data, invalid_data


def pyplot_plot_energy_invalid_vs_slices(energy_data, invalid_data, max_slices=400):
    slices_range = range(1, max_slices + 1)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    for name, energy_list in energy_data.items():
        ax1.plot(slices_range, energy_list, label=f"{name} Energy")

    ax1.set_xlabel("Number of Slices")
    ax1.set_ylabel("Total Topology Energy", color="black")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    for name, invalid_list in invalid_data.items():
        ax2.plot(slices_range, invalid_list, linestyle="--", label=f"{name} Invalids")

    ax2.set_ylabel("Invalid Placements", color="gray")
    ax2.legend(loc="upper right")

    plt.title("Energy Usage and Invalid Placements vs. Number of Slices")
    plt.tight_layout()
    plt.savefig("energy_vs_invalids.png")
    plt.close()


def plot_energy_invalid_vs_slices(energy_data, invalid_data, max_slices=400):
    slices_range = list(range(1, max_slices + 1))

    fig = go.Figure()

    # Energy Usage Lines (primary Y-axis)
    for name, energy_list in energy_data.items():
        fig.add_trace(
            go.Scatter(
                x=slices_range,
                y=energy_list,
                mode="lines",
                name=f"{name} Energy",
                yaxis="y1",
            )
        )

    # Invalid Placements Lines (secondary Y-axis)
    for name, invalid_list in invalid_data.items():
        fig.add_trace(
            go.Scatter(
                x=slices_range,
                y=invalid_list,
                mode="lines",
                name=f"{name} Invalids",
                yaxis="y2",
                line=dict(dash="dash"),
            )
        )

    # Layout with dual y-axes
    fig.update_layout(
        title="Energy Usage and Invalid Placements vs. Number of Slices",
        xaxis=dict(title="Number of Slices"),
        yaxis=dict(title="Total Topology Energy", side="left"),
        yaxis2=dict(
            title="Invalid Placements", overlaying="y", side="right", showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        width=1000,
        height=600,
    )

    # Save interactive HTML
    fig.write_html("energy_vs_invalids.html", auto_open=True)

    # Attempt PDF export (requires kaleido)
    try:
        fig.write_image("energy_vs_invalids.pdf")
    except Exception as e:
        print(f"[Warning] PDF export failed. Try: pip install kaleido\nReason: {e}")


if __name__ == "__main__":
    import sys

    topology_pickle_file = sys.argv[1]
    trained_agent_file = sys.argv[2]
    max_slices = int(sys.argv[3])

    topology_generator = NetworkTopologyGenerator(from_file=topology_pickle_file)
    topology = topology_generator.get_graph()

    state_shape = (len(topology.nodes()), len(topology.edges()))
    n_actions = len(topology.nodes())
    agent = DQNAgent(
        state_shape,
        n_actions,
        lr=0.0005,
        gamma=0.99,
        epsilon_start=0.15,
        epsilon_end=0.15,
        epsilon_decay=1.0,
        buffer_size=20000,
        batch_size=128,
        target_update=200,
        eval_mode=True,
    )
    agent.load(trained_agent_file)

    energy_data, invalid_data = evaluate_energy_vs_slices(topology, agent, max_slices)
    plot_energy_invalid_vs_slices(energy_data, invalid_data, max_slices)

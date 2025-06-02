import networkx as nx
from environment.vnf_env import VNFPlacementEnv
from environment.network_slice import NetworkSlice
from environment.barabasi_network_generator import NetworkTopologyGenerator
from main import create_sample_slices
from agents.dqn_agent import DQNAgent
import pickle
import random
import numpy as np
import copy

from topology_visualizer import TopologyVisualizer


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
        # env.step(chosen_node)
        _, reward, terminated, _, _ = env.step(chosen_node)
        if reward < 0 or (
            terminated and len(slice_obj.path) != len(slice_obj.vnf_list)
        ):
            return False

    return True


def random_valid_placement(env: VNFPlacementEnv, slice_obj: NetworkSlice) -> bool:
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
        # env.step(chosen_node)
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
            # print(f"Empty valid nodes: {valid_nodes}")
            return False

        action = agent.select_action(state, valid_nodes)
        next_state, reward, terminated, _, _ = env.step(action)
        state = next_state

        if reward < 0 or (
            terminated and len(slice_obj.path) != len(slice_obj.vnf_list)
        ):
            # print(f"Slice path: {slice_obj.path}")
            # print(f"VNFS: {slice_obj.vnf_list}")
            # print(f"lengths match: {len(slice_obj.path) == len(slice_obj.vnf_list)}")
            return False

    return True


def evaluate_all_approaches(topology, slices, agent):
    approaches = {
        "Greedy": greedy_placement,
        "Random": random_valid_placement,
        "DQN": lambda env, s: dqn_agent_placement(env, s, agent),
    }

    for name, strategy in approaches.items():
        env = VNFPlacementEnv(topology)
        env.reset()
        total_qos_violations = 0
        total_energy = 0
        total_success = 0
        done_slices = []
        for s in slices:
            clean_slice = copy.deepcopy(s)
            success = strategy(env, clean_slice)
            if success and len(clean_slice.path) == len(clean_slice.vnf_list):
                total_success += 1
                # total_energy += clean_slice.path_energy(topology)
                # print(
                #     f"Strategy: {name} - \nSlice - {clean_slice.slice_id}\nSlice Energy: {clean_slice.path_energy(topology)}"
                # )
                done_slices.append(clean_slice)
                # visualizer = TopologyVisualizer(topology)
                # visualizer.animate_slice_building(done_slices)
        total_energy = env.total_energy_used(topology)

        print(f"\n{name} Strategy")
        print(f"Success Rate: {total_success / len(slices):.2f}")
        print(f"Avg Energy per Success: {total_energy / max(total_success, 1):.2f}")
        print(f"QoS Violations: {len(slices) - total_success}")


if __name__ == "__main__":
    import sys

    topology_pickle_file = sys.argv[1]
    trained_agent_file = sys.argv[2]
    n_slices = int(sys.argv[3])

    topology_generator = NetworkTopologyGenerator(from_file=topology_pickle_file)
    topology = topology_generator.get_graph()

    # Create agent with same parameters
    state_shape = (len(topology.nodes()), len(topology.edges()))
    n_actions = len(topology.nodes())
    agent = DQNAgent(
        state_shape,
        n_actions,
        lr=0.0005,
        gamma=0.99,
        epsilon_start=0.15,  # Keep epsilon low for testing
        epsilon_end=0.15,
        epsilon_decay=1.0,  # No decay during testing
        buffer_size=20000,
        batch_size=128,
        target_update=200,
        # eval_mode=True,
    )

    # Load trained model
    agent.load(trained_agent_file)

    slices = create_sample_slices(topology, n_slices=n_slices)
    evaluate_all_approaches(topology, slices, agent)

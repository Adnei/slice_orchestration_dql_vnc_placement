from agents.dqn_agent import DQNAgent
from environment.vnf_env import VNFPlacementEnv
from environment.barabasi_network_generator import NetworkTopologyGenerator
import random
import numpy as np
import sys
from main import create_sample_slices
from topology_visualizer import TopologyVisualizer

# from main import TrainingMetrics


def test_trained_agent(topology_file, trained_agent_file):
    # Initialize the same topology
    topology_generator = NetworkTopologyGenerator(from_file=topology_file)
    topology = topology_generator.get_graph()

    # Create environment
    env = VNFPlacementEnv(topology)

    # Create agent with same parameters
    state_shape = (len(topology.nodes()), len(topology.edges()))
    n_actions = len(topology.nodes())
    agent = DQNAgent(
        state_shape,
        n_actions,
        lr=0.0005,
        gamma=0.99,
        epsilon_start=0.05,  # Keep epsilon low for testing
        epsilon_end=0.05,
        epsilon_decay=1.0,  # No decay during testing
        buffer_size=20000,
        batch_size=128,
        target_update=200,
        eval_mode=True,
    )

    # Load trained model
    agent.load(trained_agent_file)

    # Test parameters
    n_test_episodes = 1
    n_slices = 5  # Maximum difficulty

    # Metrics
    total_energy = 0
    total_success = 0

    for episode in range(n_test_episodes):
        slices = create_sample_slices(topology, n_slices=n_slices)
        episode_energy = 0
        episode_success = 0
        # Designed to allow multiple slice instantiation in the same graph
        state, _ = env.reset()
        for slice in slices:
            env.add_slice(slice)

            # Get initial valid nodes
            current_vnf_idx = 0
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

            terminated = False
            while not terminated and valid_nodes:
                action = agent.select_action(state, valid_nodes)
                next_state, reward, terminated, _, _ = env.step(action)

                state = next_state

                # Update valid nodes for next VNF
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

            # Record results
            if slice.path and len(slice.path) == len(slice.vnf_list):
                episode_success += 1
                episode_energy += slice.path_energy(topology)

            if not slice.validate_vnf_placement(topology):
                print("\n ============================================")
                print(
                    f"Slice Info: {slice}\nMeets QoS: {slice.validate_vnf_placement(topology)}"
                )
                print("\n ============================================")
        total_energy += episode_energy
        total_success += episode_success

        print(
            f"Test Episode {episode} | "
            f"Success Rate: {episode_success / n_slices:.2f} | "
            f"Avg Energy: {episode_energy / max(episode_success, 1):.2f}"
        )
        visualizer = TopologyVisualizer(topology)
        visualizer.animate_slice_building(slices)

    print("\nFinal Test Results:")
    print(f"Overall Success Rate: {total_success / (n_test_episodes * n_slices):.2f}")
    print(f"Average Energy per Slice: {total_energy / max(total_success, 1):.2f}")


if __name__ == "__main__":
    topology_pickle_file = sys.argv[1]
    trained_agent_file = sys.argv[2]
    test_trained_agent(topology_pickle_file, trained_agent_file)

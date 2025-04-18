import numpy as np
import torch
import networkx as nx
from typing import List

from ..environment.network_slice import NetworkSlice
from ..agents.dqn_agent import DQNAgent
from ..agents.baseline_agents import RandomAgent, GreedyAgent, RoundRobinAgent


class Evaluator:
    def __init__(self, topology: nx.Graph):
        self.topology = topology
        self.random_agent = RandomAgent()
        self.greedy_agent = GreedyAgent()
        self.round_robin_agent = RoundRobinAgent()

    def evaluate_model(self, agent: DQNAgent, slices: List[NetworkSlice]) -> float:
        """Evaluate DQN agent on given slices"""
        # Create a copy of the topology to avoid modifying the original
        topology_copy = self.topology.copy()
        total_energy = 0

        for slice in slices:
            # Reset slice path
            slice.path = None

            # Place slice using DQN agent
            self._place_slice_with_dqn(agent, topology_copy, slice)

            # Calculate energy consumption
            total_energy += self._calculate_energy(topology_copy)

        return total_energy

    def random_vnf_placement(self, slices: List[NetworkSlice]) -> float:
        """Random placement baseline"""
        topology_copy = self.topology.copy()
        total_energy = 0

        for slice in slices:
            if self.random_agent.place_vnfs(topology_copy, slice):
                total_energy += self._calculate_energy(topology_copy)

        return total_energy

    def greedy_vnf_placement(self, slices: List[NetworkSlice]) -> float:
        """Greedy placement baseline"""
        topology_copy = self.topology.copy()
        total_energy = 0

        for slice in slices:
            if self.greedy_agent.place_vnfs(topology_copy, slice):
                total_energy += self._calculate_energy(topology_copy)

        return total_energy

    def round_robin_vnf_placement(self, slices: List[NetworkSlice]) -> float:
        """Round-robin placement baseline"""
        topology_copy = self.topology.copy()
        total_energy = 0

        for slice in slices:
            if self.round_robin_agent.place_vnfs(topology_copy, slice):
                total_energy += self._calculate_energy(topology_copy)

        return total_energy

    def _place_slice_with_dqn(
        self, agent: DQNAgent, topology: nx.Graph, network_slice: NetworkSlice
    ) -> bool:
        """Helper method to place a slice using DQN agent"""
        # Implement slice placement logic using DQN agent
        # This would involve interacting with the environment
        # For simplicity, we're using a simplified version here

        valid_nodes = {"RAN": [], "Edge": [], "Transport": [], "Core": []}

        # Categorize nodes by type
        for node in topology.nodes():
            node_type = topology.nodes[node]["type"]
            valid_nodes[node_type].append(node)

        path = []
        for vnf in network_slice.vnf_list:
            candidates = valid_nodes[vnf.vnf_type]

            # Filter by CPU capacity
            candidates = [
                node
                for node in candidates
                if (topology.nodes[node]["cpu_usage"] + vnf.vcpu_usage)
                <= topology.nodes[node]["cpu_limit"]
            ]

            if not candidates:
                return False

            # Get state representation
            state = {
                "cpu_usage": np.array(
                    [
                        topology.nodes[node]["cpu_usage"]
                        / topology.nodes[node]["cpu_limit"]
                        for node in topology.nodes()
                    ]
                ),
                "bandwidth_usage": np.array(
                    [
                        topology.edges[edge]["link_usage"]
                        / topology.edges[edge]["link_capacity"]
                        for edge in topology.edges()
                    ]
                ),
                "current_slice": {
                    "slice_type": 0,  # Simplified
                    "qos": np.array(
                        [
                            network_slice.qos.max_latency,
                            network_slice.qos.edge_latency or 0,
                            network_slice.qos.min_bandwidth,
                        ]
                    ),
                    "vnfs_placed": len(path),
                },
            }

            # Select action
            action = agent.select_action(state, candidates)
            selected_node = candidates[action]
            path.append(selected_node)

            # Update resources
            topology.nodes[selected_node]["cpu_usage"] += vnf.vcpu_usage

        network_slice.path = path
        return True

    def _calculate_energy(self, topology: nx.Graph) -> float:
        """Calculate total energy consumption of the network"""
        total_energy = 0

        for node in topology.nodes():
            node_data = topology.nodes[node]
            total_energy += (
                node_data["energy_base"]
                + node_data["energy_per_vcpu"] * node_data["cpu_usage"]
            )

        return total_energy

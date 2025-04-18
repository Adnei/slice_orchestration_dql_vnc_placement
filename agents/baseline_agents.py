import random
import numpy as np
from typing import List
import networkx as nx

from ..environment.network_slice import NetworkSlice
from ..models.vnf import VNF


class RandomAgent:
    def place_vnfs(self, topology: nx.Graph, network_slice: NetworkSlice) -> bool:
        """Random placement respecting constraints"""
        valid_nodes = {"RAN": [], "Edge": [], "Transport": [], "Core": []}

        # Categorize nodes by type
        for node in topology.nodes():
            node_type = topology.nodes[node]["type"]
            valid_nodes[node_type].append(node)

        # Place each VNF randomly on valid nodes
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

            # Select random node
            selected_node = random.choice(candidates)
            path.append(selected_node)

            # Update resources
            topology.nodes[selected_node]["cpu_usage"] += vnf.vcpu_usage

        network_slice.path = path
        return True


class GreedyAgent:
    def place_vnfs(self, topology: nx.Graph, network_slice: NetworkSlice) -> bool:
        """Greedy placement minimizing energy consumption"""
        valid_nodes = {"RAN": [], "Edge": [], "Transport": [], "Core": []}

        # Categorize nodes by type
        for node in topology.nodes():
            node_type = topology.nodes[node]["type"]
            valid_nodes[node_type].append(node)

        # Place each VNF on node with lowest energy impact
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

            # Select node with lowest energy_per_vcpu
            selected_node = min(
                candidates, key=lambda node: topology.nodes[node]["energy_per_vcpu"]
            )
            path.append(selected_node)

            # Update resources
            topology.nodes[selected_node]["cpu_usage"] += vnf.vcpu_usage

        network_slice.path = path
        return True


class RoundRobinAgent:
    def __init__(self):
        self.node_pointers = {"RAN": 0, "Edge": 0, "Transport": 0, "Core": 0}

    def place_vnfs(self, topology: nx.Graph, network_slice: NetworkSlice) -> bool:
        """Round-robin placement across nodes"""
        valid_nodes = {"RAN": [], "Edge": [], "Transport": [], "Core": []}

        # Categorize nodes by type
        for node in topology.nodes():
            node_type = topology.nodes[node]["type"]
            valid_nodes[node_type].append(node)

        # Place each VNF in round-robin fashion
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

            # Get next node in round-robin
            ptr = self.node_pointers[vnf.vnf_type]
            selected_node = candidates[ptr % len(candidates)]
            self.node_pointers[vnf.vnf_type] = ptr + 1
            path.append(selected_node)

            # Update resources
            topology.nodes[selected_node]["cpu_usage"] += vnf.vcpu_usage

        network_slice.path = path
        return True

import numpy as np
import networkx as nx
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
import random


class VNFType(Enum):
    RAN = 0
    EDGE = 1
    TRANSPORT = 2
    CORE = 3


@dataclass
class VNF:
    vnf_id: int
    vnf_type: VNFType
    cpu_demand: int
    bandwidth_demand: int


class SimpleVNFEnv:
    def __init__(self, n_nodes=20):
        self.topology = self.create_topology(n_nodes)
        self.current_slice = None
        self.current_vnf_idx = 0

    def create_topology(self, n_nodes):
        # Simplified topology with 4 types of nodes
        G = nx.Graph()
        node_counts = {
            VNFType.RAN: int(n_nodes * 0.4),
            VNFType.EDGE: int(n_nodes * 0.3),
            VNFType.TRANSPORT: int(n_nodes * 0.2),
            VNFType.CORE: n_nodes - int(n_nodes * 0.9),
        }

        node_id = 0
        for vnf_type, count in node_counts.items():
            for _ in range(count):
                G.add_node(
                    node_id,
                    type=vnf_type,
                    cpu_capacity=100,
                    cpu_used=0,
                    energy_base=100,
                    energy_per_cpu=5,
                )
                node_id += 1

        # Add simple connectivity
        for u in G.nodes():
            for v in G.nodes():
                if u != v and random.random() < 0.1:
                    G.add_edge(u, v, latency=1, capacity=1000, used=0)

        return G

    def reset(self):
        self.current_slice = None
        self.current_vnf_idx = 0
        return self._get_state()

    def _get_state(self):
        # Enhanced state representation
        state = []

        # Current VNF information
        if self.current_slice and self.current_vnf_idx < len(self.current_slice):
            current_vnf = self.current_slice[self.current_vnf_idx]
            state.extend(
                [
                    current_vnf.vnf_type.value,
                    current_vnf.cpu_demand / 20,  # Normalized
                    current_vnf.bandwidth_demand / 50,  # Normalized
                ]
            )
        else:
            state.extend([0, 0, 0])  # Padding when no current VNF

        # Node information (top 10 most relevant nodes)
        nodes = sorted(
            self.topology.nodes(),
            key=lambda n: self.topology.nodes[n]["cpu_capacity"]
            - self.topology.nodes[n]["cpu_used"],
        )

        for node in nodes[:10]:  # Only include most available nodes
            node_data = self.topology.nodes[node]
            state.extend(
                [
                    node_data["type"].value,
                    node_data["cpu_used"] / node_data["cpu_capacity"],
                    (node_data["cpu_capacity"] - node_data["cpu_used"])
                    / 20,  # Normalized available
                    node_data["energy_per_cpu"] / 10,
                ]
            )

        return np.array(state, dtype=np.float32)

    def add_slice(self, vnf_chain: List[VNF]):
        self.current_slice = vnf_chain
        self.current_vnf_idx = 0

    def step(self, action):
        if not self.current_slice:
            return self._get_state(), 0, True, {}

        current_vnf = self.current_slice[self.current_vnf_idx]
        node_data = self.topology.nodes[action]

        # Validate action
        if (
            node_data["type"] != current_vnf.vnf_type
            or node_data["cpu_used"] + current_vnf.cpu_demand
            > node_data["cpu_capacity"]
        ):
            # Large penalty for invalid action
            return self._get_state(), -5, True, {}

        # Place VNF
        node_data["cpu_used"] += current_vnf.cpu_demand
        self.current_vnf_idx += 1

        # Calculate reward components
        placement_reward = 1.0  # Base reward for successful placement
        energy_penalty = (
            node_data["energy_base"]
            + node_data["energy_per_cpu"] * current_vnf.cpu_demand
        ) / 1000
        completion_bonus = 5.0 if self.current_vnf_idx >= len(self.current_slice) else 0

        reward = placement_reward - energy_penalty + completion_bonus

        # Check if done
        done = self.current_vnf_idx >= len(self.current_slice)
        return self._get_state(), reward, done, {}

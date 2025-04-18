import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from environment.network_slice import NetworkSlice, SliceType, VNF


class VNFPlacementEnv(gym.Env):
    def __init__(self, topology: nx.Graph, max_slices: int = 10):
        super().__init__()
        self.topology = topology
        self.max_slices = max_slices
        self.current_slices: List[NetworkSlice] = []

        # Calculate maximum possible energy per step
        self.max_energy_per_step = self._calculate_max_energy_step()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(topology.nodes()))
        self.observation_space = spaces.Dict(
            {
                "cpu_usage": spaces.Box(low=0, high=1, shape=(len(topology.nodes()),)),
                "bandwidth_usage": spaces.Box(
                    low=0, high=1, shape=(len(topology.edges()),)
                ),
                "current_slice": spaces.Dict(
                    {
                        "slice_type": spaces.Discrete(len(SliceType)),
                        "qos": spaces.Box(low=0, high=np.inf, shape=(3,)),
                        "vnfs_placed": spaces.Discrete(10),
                    }
                ),
            }
        )

        self.reset()

    def _calculate_max_energy_step(self) -> float:
        """Calculate maximum possible energy increase from a single placement"""
        max_energy = max(
            data["energy_per_vcpu"] * data["cpu_limit"]
            for _, data in self.topology.nodes(data=True)
        )
        return max_energy * 1.1  # Add 10% margin

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_slices = []
        self._update_resource_usage()
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Place next VNF in the current slice"""
        reward = 0
        terminated = False
        truncated = False
        current_slice = self.current_slices[-1] if self.current_slices else None

        if current_slice:
            current_vnf_idx = len(current_slice.path or [])
            if current_vnf_idx >= len(current_slice.vnf_list):
                return self._get_observation(), 0, True, False, {}

            current_vnf = current_slice.vnf_list[current_vnf_idx]
            target_node = list(self.topology.nodes())[action]

            if self._validate_placement(target_node, current_vnf):
                # Valid placement
                energy_increase = (
                    self.topology.nodes[target_node]["energy_per_vcpu"]
                    * current_vnf.vcpu_usage
                )
                reward = -energy_increase

                # Update resources
                self.topology.nodes[target_node]["cpu_usage"] += current_vnf.vcpu_usage
                if not current_slice.path:
                    current_slice.path = [target_node]
                else:
                    current_slice.path.append(target_node)

                # Check if slice placement is complete
                if len(current_slice.path) == len(current_slice.vnf_list):
                    terminated = True
                    if current_slice.validate_vnf_placement(self.topology):
                        reward += 10  # Bonus for QoS compliance
            else:
                # Invalid placement - terminate episode with penalty
                reward = -self.max_energy_per_step
                terminated = True

        return self._get_observation(), reward, terminated, truncated, {}

    def _validate_placement(self, node: int, vnf: VNF) -> bool:
        """Check if VNF can be placed on node"""
        node_data = self.topology.nodes[node]

        # Check type compatibility
        if vnf.vnf_type != node_data["type"]:
            return False

        # Check CPU capacity
        if (node_data["cpu_usage"] + vnf.vcpu_usage) > node_data["cpu_limit"]:
            return False

        # Check path continuity for slice
        if self.current_slices and self.current_slices[-1].path:
            last_node = self.current_slices[-1].path[-1]
            if not self.topology.has_edge(last_node, node):
                return False

        return True

    def _get_observation(self) -> Dict:
        """Return current state observation"""
        # Normalize resource usage
        cpu_usage = np.array(
            [
                self.topology.nodes[node]["cpu_usage"]
                / self.topology.nodes[node]["cpu_limit"]
                for node in self.topology.nodes()
            ]
        )

        bandwidth_usage = np.array(
            [
                self.topology.edges[edge]["link_usage"]
                / self.topology.edges[edge]["link_capacity"]
                for edge in self.topology.edges()
            ]
        )

        # Current slice info
        current_slice_info = {"slice_type": 0, "qos": np.zeros(3), "vnfs_placed": 0}

        if self.current_slices:
            current_slice = self.current_slices[-1]
            current_slice_info = {
                "slice_type": list(SliceType).index(current_slice.slice_type),
                "qos": np.array(
                    [
                        current_slice.qos.max_latency,
                        current_slice.qos.edge_latency or 0,
                        current_slice.qos.min_bandwidth,
                    ]
                ),
                "vnfs_placed": len(current_slice.path or []),
            }

        return {
            "cpu_usage": cpu_usage,
            "bandwidth_usage": bandwidth_usage,
            "current_slice": current_slice_info,
        }

    def _update_resource_usage(self):
        """Update link bandwidth usage based on placed slices"""
        # Reset all link usages
        for edge in self.topology.edges():
            self.topology.edges[edge]["link_usage"] = 0

        # Update based on active slices
        for slice in self.current_slices:
            if slice.path:
                for i in range(len(slice.path) - 1):
                    u, v = slice.path[i], slice.path[i + 1]
                    if self.topology.has_edge(u, v):
                        self.topology.edges[u, v]["link_usage"] += sum(
                            vnf.bandwidth_usage for vnf in slice.vnf_list
                        )

    def add_slice(self, network_slice: NetworkSlice):
        """Add a new slice to be placed"""
        if len(self.current_slices) >= self.max_slices:
            raise ValueError("Maximum number of slices reached")
        self.current_slices.append(network_slice)

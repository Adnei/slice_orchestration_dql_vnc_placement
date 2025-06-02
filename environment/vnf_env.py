import numpy as np
import networkx as nx
from typing import Dict, Tuple, Union
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from environment.network_slice import NetworkSlice, SliceType, VNF


class VNFPlacementEnv(gym.Env):
    def __init__(self, topology: nx.Graph, max_slices: int = 9999999):
        super().__init__()
        self.topology = topology
        self.max_slices = max_slices
        self.accumulated_latency = 0.0
        self.current_slices: List[NetworkSlice] = []
        self.slice_type_mapping = {
            SliceType.URLLC: 0,
            SliceType.EMBB: 1,
            SliceType.MMTC: 2,
            SliceType.GENERIC: 3,
        }

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

        # Clear CPU usage and hosted VNFs from all nodes
        for node in self.topology.nodes:
            self.topology.nodes[node]["cpu_usage"] = 0
            self.topology.nodes[node]["hosted_vnfs"] = []

        # Clear all bandwidth usage on links
        for edge in self.topology.edges:
            self.topology.edges[edge]["link_usage"] = 0

        # Reset slice list
        self.current_slices = []

        return self._get_observation(), {}

    def step(self, action):
        if not self.current_slices:
            return self._get_observation(), 0.0, True, False, {}

        current_slice = self.current_slices[-1]
        current_vnf_idx = len(current_slice.path) if current_slice.path else 0

        # Handle invalid action
        if action == -1:
            return self._get_observation(), -1000.0, True, False, {}

        target_node = list(self.topology.nodes())[action]
        prev_node = None if not current_slice.path else current_slice.path[-1]
        current_vnf = current_slice.vnf_list[current_vnf_idx]

        # Validate placement (different than QoS)
        if not self._validate_placement(prev_node, target_node, current_vnf):
            return self._get_observation(), -1000.0, True, False, {}

        prev_energy = self.total_energy_used(self.topology)

        # Place VNF
        is_node_active = self.topology.nodes[target_node]["cpu_usage"] > 0
        self._update_resource_usage(prev_node, target_node, current_vnf)
        current_slice.add_vnf_node(target_node)
        qos_met = current_slice.validate_vnf_placement(self.topology)
        if not qos_met:
            return self._get_observation(), -1000, True, False, {}

        # Path quality
        # --> Se conseguir avaliar a qualidade do caminho, em termos de "esse caminho é promissor", melhora MUITO a recompensa
        # --> É necessário identificar alguma propriedade que identifique que um determinado caminho é promissor. Seria interessante verificar teoria das redes (grafos) para buscar algo que ajude nisso
        #   --> Por exemplo, grau do nó, muitos recursos disponíveis, nó centralizado, etc... alguma heuristica que permita evitar o calculo completo do shortest_path
        # Path quality desativado até que isso seja resolvido
        # path_quality = 0.5 if current_slice.path else 0.0

        new_energy = self.total_energy_used(self.topology)
        delta_energy = new_energy - prev_energy
        delta_energy *= 2 if is_node_active else 1
        latency_penalty = 0.5 * current_slice.path_latency(self.topology)
        link_bonus = (
            0
            if not prev_node
            else 0.1 * current_slice.path_available_bandwidth(self.topology)
        )

        reuse_bonus = 50 if is_node_active else 0  # +20 for reusing a powered node
        activation_penalty = 50 if not is_node_active else 0

        reward = (
            100.0
            - delta_energy / 100
            - latency_penalty
            + 0.1 * link_bonus / 1000
            + reuse_bonus
            - activation_penalty
        )

        # Completion bonus
        if len(current_slice.path) == len(current_slice.vnf_list):
            reward += 100  # equivalent to --> 100_000 / 100

            # print(
            #    f"[QoS Check] Type: {current_slice.slice_type} | Max Latency: {current_slice.qos.max_latency} | Min Bandwidth: {current_slice.qos.min_bandwidth} --------- "
            #    f"Actual: {current_slice.path_latency(self.topology):.2f} | Bandwidth OK: {current_slice.path_available_bandwidth(self.topology)}"
            # )

            return self._get_observation(), reward, True, False, {}

        return self._get_observation(), reward, False, False, {}

    def _validate_placement(self, prev_node: int | None, node: int, vnf: VNF) -> bool:
        """Check if VNF can be placed on node"""
        node_data = self.topology.nodes[node]

        # Check type compatibility
        if vnf.vnf_type != node_data["type"]:
            return False

        # Check CPU capacity
        if (node_data["cpu_usage"] + vnf.vcpu_usage) > node_data["cpu_limit"]:
            return False

        # Greedy --> Could check a path to this node
        if prev_node and not self.topology.has_edge(prev_node, node):
            return False

        return True

    def _get_observation(self) -> Dict[str, Union[np.ndarray, Dict]]:
        """Returns the current environment observation as a structured dictionary.

        Ensures:
        - All arrays are float32 for PyTorch compatibility
        - Values are properly normalized [0,1] where applicable
        - Consistent structure for both single and batched processing
        """
        # Normalized CPU usage (0-1 scale)
        cpu_usage = np.zeros(len(self.topology.nodes), dtype=np.float32)
        for i, node in enumerate(self.topology.nodes()):
            cpu_capacity = self.topology.nodes[node]["cpu_limit"]
            cpu_usage[i] = self.topology.nodes[node]["cpu_usage"] / max(
                cpu_capacity, 1e-6
            )  # Avoid division by zero

        # Normalized bandwidth usage (0-1 scale)
        bandwidth_usage = np.zeros(len(self.topology.edges()), dtype=np.float32)
        for i, edge in enumerate(self.topology.edges()):
            link_capacity = self.topology.edges[edge]["link_capacity"]
            bandwidth_usage[i] = self.topology.edges[edge]["link_usage"] / max(
                link_capacity, 1e-6
            )

        # Current slice information
        if self.current_slices:
            current_slice = self.current_slices[-1]
            slice_info = {
                "slice_type": self._slice_type_to_int(current_slice.slice_type),
                "qos": np.array(
                    [
                        current_slice.qos.max_latency,
                        current_slice.qos.edge_latency or 0.0,
                        current_slice.qos.min_bandwidth,
                    ],
                    dtype=np.float32,
                ),
                "vnfs_placed": len(current_slice.path) if current_slice.path else 0,
            }
        else:
            # Default values when no active slice
            slice_info = {
                "slice_type": 0,
                "qos": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "vnfs_placed": 0,
            }

        return {
            "cpu_usage": cpu_usage,
            "bandwidth_usage": bandwidth_usage,
            "current_slice": slice_info,
        }

    def _slice_type_to_int(self, slice_type: SliceType) -> int:
        """Convert SliceType enum to integer representation"""
        return self.slice_type_mapping[slice_type]

    def _int_to_slice_type(self, val: int) -> SliceType:
        """Reverse mapping"""
        return [SliceType.URLLC, SliceType.EMBB, SliceType.MMTC][val]

    def _update_resource_usage(self, prev_node, target_node, current_vnf):
        """Update link bandwidth, cpu usage, and hosted vnfs"""
        if prev_node:
            self.topology.edges[prev_node, target_node]["link_usage"] += (
                current_vnf.bandwidth_usage
            )
        self.topology.nodes[target_node]["cpu_usage"] += current_vnf.vcpu_usage
        self.topology.nodes[target_node]["hosted_vnfs"].append(current_vnf)

    def add_slice(self, network_slice: NetworkSlice):
        """Add a new slice to be placed"""
        if len(self.current_slices) >= self.max_slices:
            raise ValueError("Maximum number of slices reached")
        self.current_slices.append(network_slice)

    def get_observation(self):
        return self._get_observation()

    def total_energy_used(self, topology: nx.Graph):
        total = 0
        for node in topology.nodes:
            node_data = topology.nodes[node]
            if node_data["cpu_usage"] > 0:
                total += node_data["energy_base"]
            total += node_data["cpu_usage"] * node_data["energy_per_vcpu"]
        return total

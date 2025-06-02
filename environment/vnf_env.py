import numpy as np
import networkx as nx
from typing import Dict, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
from environment.network_slice import NetworkSlice, SliceType, VNF


class VNFPlacementEnv(gym.Env):
    def __init__(self, topology: nx.Graph, max_slices: int = 9999999):
        super().__init__()
        self.topology = topology
        self.max_slices = max_slices
        self.accumulated_latency = 0.0
        self.current_slices: list[NetworkSlice] = []
        self.slice_type_mapping = {
            SliceType.URLLC: 0,
            SliceType.EMBB: 1,
            SliceType.MMTC: 2,
            SliceType.GENERIC: 3,
        }

        self.prev_node_activation = {node: False for node in topology.nodes()}

        self.action_space = spaces.Discrete(len(topology.nodes()))
        self.observation_space = spaces.Dict(
            {
                "cpu_usage": spaces.Box(
                    low=0, high=1, shape=(len(topology.nodes()),), dtype=np.float32
                ),
                "bandwidth_usage": spaces.Box(
                    low=0, high=1, shape=(len(topology.edges()),), dtype=np.float32
                ),
                "node_on": spaces.MultiBinary(len(topology.nodes())),
                "energy_base": spaces.Box(
                    low=0, high=np.inf, shape=(len(topology.nodes()),), dtype=np.float32
                ),
                "current_slice": spaces.Dict(
                    {
                        "slice_type": spaces.Discrete(len(SliceType)),
                        "qos": spaces.Box(
                            low=0, high=np.inf, shape=(3,), dtype=np.float32
                        ),
                        "vnfs_placed": spaces.Discrete(10),
                    }
                ),
            }
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for node in self.topology.nodes:
            self.topology.nodes[node]["cpu_usage"] = 0
            self.topology.nodes[node]["hosted_vnfs"] = []

        for edge in self.topology.edges:
            self.topology.edges[edge]["link_usage"] = 0

        self.prev_node_activation = {node: False for node in self.topology.nodes()}
        self.current_slices = []

        return self._get_observation(), {}

    def step(self, action):
        if not self.current_slices:
            return self._get_observation(), 0.0, True, False, {}

        current_slice = self.current_slices[-1]
        current_vnf_idx = len(current_slice.path) if current_slice.path else 0

        if action == -1:
            return self._get_observation(), -1000.0, True, False, {}

        target_node = list(self.topology.nodes())[action]
        prev_node = None if not current_slice.path else current_slice.path[-1]
        current_vnf = current_slice.vnf_list[current_vnf_idx]

        if not self._validate_placement(prev_node, target_node, current_vnf):
            return self._get_observation(), -1000.0, True, False, {}

        was_node_active = self.topology.nodes[target_node]["cpu_usage"] > 0
        prev_energy = self.total_energy_used()

        self._update_resource_usage(prev_node, target_node, current_vnf)
        current_slice.add_vnf_node(target_node)

        qos_met = current_slice.validate_vnf_placement(self.topology)
        if not qos_met:
            return self._get_observation(), -1000, True, False, {}

        new_energy = self.total_energy_used()
        delta_energy = new_energy - prev_energy

        latency_penalty = 0.5 * current_slice.path_latency(self.topology)
        link_bonus = (
            0
            if not prev_node
            else 0.1 * current_slice.path_available_bandwidth(self.topology)
        )

        # Dynamic reward shaping based on energy base
        energy_base = self.topology.nodes[target_node]["energy_base"]
        reuse_bonus = 0.2 * energy_base if was_node_active else 0
        activation_penalty = 0.25 * energy_base if not was_node_active else 0

        reward = (
            100.0
            - delta_energy / 100
            - latency_penalty
            + 0.1 * link_bonus / 1000
            + reuse_bonus
            - activation_penalty
        )

        if len(current_slice.path) == len(current_slice.vnf_list):
            reward += 100
            return self._get_observation(), reward, True, False, {}

        return self._get_observation(), reward, False, False, {}

    def _validate_placement(self, prev_node: int | None, node: int, vnf: VNF) -> bool:
        node_data = self.topology.nodes[node]
        if vnf.vnf_type != node_data["type"]:
            return False
        if (node_data["cpu_usage"] + vnf.vcpu_usage) > node_data["cpu_limit"]:
            return False
        if prev_node and not self.topology.has_edge(prev_node, node):
            return False
        return True

    def _get_observation(self) -> Dict[str, Union[np.ndarray, Dict]]:
        cpu_usage = np.zeros(len(self.topology.nodes), dtype=np.float32)
        node_on = np.zeros(len(self.topology.nodes), dtype=np.int8)
        energy_base = np.zeros(len(self.topology.nodes), dtype=np.float32)

        for i, node in enumerate(self.topology.nodes()):
            data = self.topology.nodes[node]
            cpu_usage[i] = data["cpu_usage"] / max(data["cpu_limit"], 1e-6)
            node_on[i] = int(data["cpu_usage"] > 0)
            energy_base[i] = data["energy_base"]

        bandwidth_usage = np.zeros(len(self.topology.edges()), dtype=np.float32)
        for i, edge in enumerate(self.topology.edges()):
            usage = self.topology.edges[edge]["link_usage"]
            capacity = self.topology.edges[edge]["link_capacity"]
            bandwidth_usage[i] = usage / max(capacity, 1e-6)

        if self.current_slices:
            s = self.current_slices[-1]
            slice_info = {
                "slice_type": self.slice_type_mapping[s.slice_type],
                "qos": np.array(
                    [
                        s.qos.max_latency,
                        s.qos.edge_latency or 0.0,
                        s.qos.min_bandwidth,
                    ],
                    dtype=np.float32,
                ),
                "vnfs_placed": len(s.path) if s.path else 0,
            }
        else:
            slice_info = {
                "slice_type": 0,
                "qos": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "vnfs_placed": 0,
            }

        return {
            "cpu_usage": cpu_usage,
            "bandwidth_usage": bandwidth_usage,
            "node_on": node_on,
            "energy_base": energy_base,
            "current_slice": slice_info,
        }

    def _update_resource_usage(self, prev_node, target_node, vnf):
        if prev_node:
            self.topology.edges[prev_node, target_node]["link_usage"] += (
                vnf.bandwidth_usage
            )
        self.topology.nodes[target_node]["cpu_usage"] += vnf.vcpu_usage
        self.topology.nodes[target_node]["hosted_vnfs"].append(vnf)

    def add_slice(self, network_slice: NetworkSlice):
        if len(self.current_slices) >= self.max_slices:
            raise ValueError("Maximum number of slices reached")
        self.current_slices.append(network_slice)

    def get_observation(self):
        return self._get_observation()

    def total_energy_used(self, topology=None):
        topology = topology or self.topology
        total = 0
        for node in topology.nodes:
            data = topology.nodes[node]
            if data["cpu_usage"] > 0:
                total += data["energy_base"]
            total += data["cpu_usage"] * data["energy_per_vcpu"]
        return total

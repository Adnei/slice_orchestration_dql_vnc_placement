from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from numpy import double
import networkx as nx


#     URLLC = "URLLC"  # Ultra-Reliable Low Latency Communications
#     EMBB = "eMBB"  # Enhanced Mobile Broadband
#     MMTC = "mMTC"  # Massive Machine Type Communications
#     GENERIC = "GENERIC"
class SliceType(Enum):
    URLLC = auto()
    EMBB = auto()
    MMTC = auto()
    GENERIC = auto()


@dataclass
class QoS:
    qos_id: int
    max_latency: float  # ms, end-to-end
    min_bandwidth: float  # Mbps
    edge_latency: Optional[float] = None  # ms, for URLLC edge processing


@dataclass
class VNF:
    vnf_id: int
    delay: float  # ms (0.1-0.5)
    vnf_type: str  # "RAN", "Edge", "Transport", "Core"
    vcpu_usage: int
    bandwidth_usage: float  # Mbps

    def __post_init__(self):
        if self.delay < 0.1 or self.delay > 0.5:
            raise ValueError("VNF delay must be between 0.1ms and 0.5ms")
        if self.vnf_type not in ["RAN", "Edge", "Transport", "Core"]:
            raise ValueError("Invalid VNF type")


class NetworkSlice:
    def __init__(self, slice_id: int, slice_type: SliceType, qos: QoS, origin: int):
        self.slice_id = slice_id
        self.slice_type = slice_type
        self.qos = qos
        self.origin = origin  # RAN node ID
        self.vnf_list: List[VNF] = []
        self.path: List[int] = []
        self.instantiated = False

    def __str__(self):
        return f"Slice ID: {self.slice_id}\nSlice Path:{self.path}"

    def path_latency(self, topology: "nx.Graph"):
        total_latency = 0
        # Calculate path latency
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i + 1]
            total_latency += topology.edges[u, v]["latency"]

        return total_latency

    def path_available_bandwidth(self, topology: "nx.Graph"):
        available_bandwidth = np.inf
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i + 1]
            available = (
                topology.edges[u, v]["link_capacity"]
                - topology.edges[u, v]["link_usage"]
            )
            available_bandwidth = min(available_bandwidth, available)
        return available_bandwidth

    def path_energy(self, topology: "nx.Graph"):
        energy = 0
        for node in self.path:
            energy += (
                topology.nodes[node]["energy_base"]
                + topology.nodes[node]["cpu_usage"]
                * topology.nodes[node]["energy_per_vcpu"]
            )
        return energy

    def add_vnf(self, vnf: VNF):
        """Add a VNF to the slice's function chain"""
        self.vnf_list.append(vnf)

    def add_vnf_node(self, node):
        self.path.append(node)

    def validate_vnf_placement(self, topology: "nx.Graph") -> bool:
        """Check if current path meets QoS requirements"""
        if not self.path:
            return False

        total_latency = self.path_latency(topology)
        available_bandwidth = self.path_available_bandwidth(topology)

        # Add VNF processing delays
        total_latency += sum(vnf.delay for vnf in self.vnf_list)

        # @TODO Check edge latency for URLLC
        # Check edge latency for URLLC
        # if self.slice_type == SliceType.URLLC and self.qos.edge_latency:
        #     edge_nodes = [
        #         node for node in self.path if topology.nodes[node]["type"] == "Edge"
        #     ]
        #     if not edge_nodes:
        #         return False
        #     # Simplified edge latency check
        #     if (
        #         topology.nodes[edge_nodes[0]]["processing_delay"]
        #         > self.qos.edge_latency
        #     ):
        #         return False

        # Check bandwidth (simplified - sum of all VNF bandwidths)
        # total_bandwidth = sum(vnf.bandwidth_usage for vnf in self.vnf_list)

        return (
            total_latency <= self.qos.max_latency
            and available_bandwidth >= self.qos.min_bandwidth
        )

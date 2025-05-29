import networkx as nx
import random
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List
import numpy as np


class NetworkTopologyGenerator:
    def __init__(self, from_file=None, n_nodes=50, avg_degree=4, rewire_prob=0.1):
        self.n_nodes = n_nodes
        self.avg_degree = avg_degree
        self.rewire_prob = rewire_prob
        self.node_types = {}
        self.node_mapping = {}  # Maps integer IDs to hierarchical names

        if type(from_file) == str:
            self.graph = self.import_graph_from_pickle(from_file)
            # Initialize node_types from loaded graph
            self.node_types = {
                node: data["type"] for node, data in self.graph.nodes(data=True)
            }
            print(f"Graph loaded from file: {from_file}")
        else:
            # Create new hierarchical topology
            self.graph = nx.Graph()
            self._create_hierarchical_topology()
            self._add_node_attributes()
            self._add_edge_attributes()
            self._ensure_connectivity_rules()

    def _create_hierarchical_topology(self):
        """Create hierarchical BA network with integer IDs"""
        node_id = 0
        layers = {
            "RAN": int(self.n_nodes * 0.4),
            "Edge": int(self.n_nodes * 0.3),
            "Transport": int(self.n_nodes * 0.2),
            "Core": self.n_nodes - int(self.n_nodes * 0.9),  # Remaining 10%
        }

        # Create nodes with integer IDs
        for layer, count in layers.items():
            for i in range(count):
                self.graph.add_node(node_id, type=layer)
                self.node_mapping[node_id] = f"{layer}_{i}"
                self.node_types[node_id] = layer
                node_id += 1

        # Create BA connections within layers
        self._connect_within_layers()

        # Connect between layers
        self._connect_between_layers()

    def _connect_within_layers(self):
        """BA connections within each layer"""
        m_params = {"RAN": 1, "Edge": 2, "Transport": 3, "Core": 4}

        for layer, m in m_params.items():
            layer_nodes = [
                n for n in self.graph.nodes if self.graph.nodes[n]["type"] == layer
            ]
            if len(layer_nodes) <= m:
                continue

            ba = nx.barabasi_albert_graph(len(layer_nodes), m)
            mapping = {i: layer_nodes[i] for i in range(len(layer_nodes))}
            ba = nx.relabel_nodes(ba, mapping)
            self.graph.add_edges_from(ba.edges())

    def _connect_between_layers(self):
        """Hierarchical inter-layer connections"""
        ran_nodes = [
            n for n in self.graph.nodes if self.graph.nodes[n]["type"] == "RAN"
        ]
        edge_nodes = [
            n for n in self.graph.nodes if self.graph.nodes[n]["type"] == "Edge"
        ]
        transport_nodes = [
            n for n in self.graph.nodes if self.graph.nodes[n]["type"] == "Transport"
        ]
        core_nodes = [
            n for n in self.graph.nodes if self.graph.nodes[n]["type"] == "Core"
        ]

        # RAN → Edge (1-2 links per RAN)
        for ran in ran_nodes:
            targets = random.sample(edge_nodes, k=random.randint(1, 2))
            for edge in targets:
                self.graph.add_edge(ran, edge)

        # Edge → Transport (2-3 links per Edge)
        for edge in edge_nodes:
            targets = random.sample(transport_nodes, k=random.randint(2, 3))
            for trans in targets:
                self.graph.add_edge(edge, trans)

        # Transport ↔ Core (full mesh)
        for trans in transport_nodes:
            for core in core_nodes:
                self.graph.add_edge(trans, core)

        # Core (full mesh)
        for i, c1 in enumerate(core_nodes):
            for c2 in core_nodes[i + 1 :]:
                self.graph.add_edge(c1, c2)

    def _ensure_connectivity_rules(self):
        """Enforce 5G connectivity constraints"""
        # Remove any invalid connections
        for u, v in list(self.graph.edges()):
            u_type = self.graph.nodes[u]["type"]
            v_type = self.graph.nodes[v]["type"]

            # No direct RAN-Transport
            if {"RAN", "Transport"} == {u_type, v_type}:
                self.graph.remove_edge(u, v)

            # No direct RAN-Core
            if {"RAN", "Core"} == {u_type, v_type}:
                self.graph.remove_edge(u, v)

            # No direct Edge-Core
            if {"Edge", "Core"} == {u_type, v_type}:
                self.graph.remove_edge(u, v)

    def _add_node_attributes(self):
        """Validated 5G node attributes based on 3GPP TR 38.801/ETSI NFV"""
        profiles = {
            "RAN": {  # O-RAN Distributed Unit
                "cpu_limit": random.randint(32, 64),
                "memory_gb": random.randint(32, 64),
                "energy_base": random.uniform(150, 250),  # Watts (Nokia AirScale)
                "energy_per_vcpu": random.uniform(2, 5),
                "processing_delay": random.uniform(0.1, 0.2),  # 150μs (TS 38.801)
                "hosted_vnfs": [],
                "cpu_usage": 0,
            },
            "Edge": {  # MEC Server
                "cpu_limit": random.randint(32, 128),  # Dell EMC PowerEdge XR4000
                "memory_gb": random.randint(32, 256),
                "energy_base": random.uniform(300, 500),
                "energy_per_vcpu": random.uniform(4, 6),
                "processing_delay": random.uniform(0.3, 0.4),  # 400μs (ETSI MEC 003)
                "hosted_vnfs": [],
                "cpu_usage": 0,
            },
            "Transport": {  # Regional DC
                "cpu_limit": random.randint(128, 256),  # Cisco UCS C480
                "memory_gb": random.randint(256, 512),
                "energy_base": random.uniform(800, 1000),
                "energy_per_vcpu": random.uniform(8, 14),
                "processing_delay": random.uniform(0.65, 0.75),  # 750μs
                "hosted_vnfs": [],
                "cpu_usage": 0,
            },
            "Core": {  # 5G Core
                "cpu_limit": random.randint(256, 512),  # NVIDIA HGX A100
                "memory_gb": random.randint(512, 1024),
                "energy_base": random.uniform(1600, 1800),
                "energy_per_vcpu": random.uniform(14, 18),
                "processing_delay": random.uniform(1, 1.5),  # 1.5ms
                "hosted_vnfs": [],
                "cpu_usage": 0,
            },
        }
        for node in self.graph.nodes:
            self.graph.nodes[node].update(profiles[self.graph.nodes[node]["type"]])

    def _add_edge_attributes(self):
        """Validated latency/capacity from 3GPP TS 38.104/ETSI GS NFV-INF 001"""
        for u, v in self.graph.edges():
            u_type = self.graph.nodes[u]["type"]
            v_type = self.graph.nodes[v]["type"]

            if "RAN" in {u_type, v_type}:  # Fronthaul
                self.graph.edges[u, v].update(
                    {
                        "latency": 0.5,  # CPRI/eCPRI target
                        "link_capacity": 10e3,  # 10Gbps
                        "link_usage": 0,
                    }
                )
            elif "Edge" in {u_type, v_type}:  # Midhaul
                self.graph.edges[u, v].update(
                    {"latency": 0.8, "link_capacity": 25e3, "link_usage": 0}  # 25Gbps
                )
            else:  # Backhaul/Core
                self.graph.edges[u, v].update(
                    {"latency": 0.3, "link_capacity": 100e3, "link_usage": 0}  # 100Gbps
                )

    # Utility Methods (unchanged from your original)
    def get_graph(self):
        return self.graph

    def draw(self, file_name="topology.pdf"):
        color_map = {
            "RAN": "red",
            "Edge": "blue",
            "Transport": "green",
            "Core": "purple",
        }
        colors = [
            color_map[self.graph.nodes[node]["type"]] for node in self.graph.nodes()
        ]

        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 12))
        nx.draw(
            self.graph,
            pos,
            node_color=colors,
            with_labels=True,
            labels=self.node_mapping,
            node_size=500,
            font_size=8,
            font_weight="bold",
        )
        plt.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=type,
                )
                for type, color in color_map.items()
            ],
            loc="upper right",
        )
        plt.title("5G Network Topology")
        plt.savefig(file_name)
        plt.close()

    def export_graph_to_pickle(self, filename="5G_Network_Topology.pickle"):
        pickle.dump(self.graph, open(filename, "wb"))

    def import_graph_from_pickle(self, filename="5G_Network_Topology.pickle"):
        return pickle.load(open(filename, "rb"))

    def get_node_name(self, node_id: int) -> str:
        """Get hierarchical name for a node"""
        return self.node_mapping.get(node_id, str(node_id))

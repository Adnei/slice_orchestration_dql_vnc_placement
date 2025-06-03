import plotly.graph_objects as go
import networkx as nx
import random
import time


class TopologyVisualizer:
    def __init__(self, topology: nx.Graph):
        self.topology = topology
        self.pos = nx.spring_layout(topology, seed=42)
        self.slice_colors = {}

    def _get_node_trace(self):
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            hovertext=[],
            mode="markers+text",
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                showscale=False,
                color=[],
                size=20,
                line_width=2,
            ),
        )

        for node in self.topology.nodes():
            x, y = self.pos[node]
            node_type = self.topology.nodes[node].get("type", "Unknown")
            cpu_usage = self.topology.nodes[node].get("cpu_usage", "N/A")
            cpu_limit = self.topology.nodes[node].get("cpu_limit", "N/A")
            energy_base = self.topology.nodes[node].get("energy_base", "N/A")
            hosted_vnfs = self.topology.nodes[node].get("hosted_vnfs", [])
            energy_per_vcpu = self.topology.nodes[node].get("energy_per_vcpu", "N/A")
            total_energy = energy_base + energy_per_vcpu * cpu_usage

            # Short label for visibility
            node_trace["text"] += (f"{node} ({node_type})",)
            node_trace["x"] += (x,)
            node_trace["y"] += (y,)

            # Detailed hover text
            node_trace["hovertext"] += (
                f"<b>Node {node} ({node_type})</b><br>"
                f"<b>Total Energy: {total_energy:.2f}</b><br>"
                f"CPU Usage: {cpu_usage}/{cpu_limit}<br>"
                f"Energy Base: {energy_base:.2f}<br>"
                f"Energy/VCPU: {energy_per_vcpu:.2f}<br>"
                f"Hosting VNFs: {len(hosted_vnfs)}",
            )

            color = {
                "RAN": "blue",
                "Edge": "orange",
                "Transport": "green",
                "Core": "red",
            }.get(node_type, "gray")
            node_trace["marker"]["color"] += (color,)

        return node_trace

    def _get_edge_trace(self):
        edge_lines = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(width=1.5, color="#888"),
            hoverinfo="none",  # disable hover for base lines
        )

        edge_hover = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)"),  # fully transparent
            hoverinfo="text",
            text=[],
        )

        for edge in self.topology.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]

            # Add the edge line
            edge_lines["x"] += (x0, x1, None)
            edge_lines["y"] += (y0, y1, None)

            # Add invisible midpoint marker for hover
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            bw = self.topology.edges[edge].get("link_capacity", "N/A")
            latency = self.topology.edges[edge].get("latency", "N/A")
            used_bw = self.topology.edges[edge].get("link_usage", "0")
            usage_percent = (used_bw / bw) * 100

            edge_hover["x"] += (mx,)
            edge_hover["y"] += (my,)
            edge_hover["text"] += (
                f"<b>Link {edge[0]} ↔ {edge[1]}</b><br>Latency: {latency}<br>"
                f"Usage (%): {usage_percent:.2f}",
            )

        return [edge_lines, edge_hover]

    def _get_slice_trace(self, slice_obj, slice_id):
        path = slice_obj.path
        color = self.slice_colors.get(slice_id)
        if not color:
            color = f"rgba({random.randint(50, 255)}, {random.randint(50, 255)}, {random.randint(50, 255)}, 0.8)"
            self.slice_colors[slice_id] = color

        trace = go.Scatter(
            x=[self.pos[node][0] for node in path],
            y=[self.pos[node][1] for node in path],
            mode="lines+markers",
            marker=dict(size=14, color=color, line=dict(width=2, color="black")),
            line=dict(width=4, color=color),
            name=f"Slice {slice_id} ({slice_obj.slice_type.name})",
            text=[f"Slice {slice_id} - VNF {i}" for i in range(len(path))],
            hoverinfo="text",
        )
        return trace

    def animate_slice_building(self, slices, delay=1.5, complete_fig_name=""):
        # fig = go.Figure()
        topology_consumption = self._get_topology_consumption()
        fig = None
        for i, slice_obj in enumerate(slices):
            fig = go.Figure()
            # fig.add_trace(self._get_edge_trace())
            for trace in self._get_edge_trace():
                fig.add_trace(trace)
            fig.add_trace(self._get_node_trace())

            for j in range(i + 1):
                if slices[j].path:
                    trace = self._get_slice_trace(slices[j], slices[j].slice_id)
                    fig.add_trace(trace)

            fig.update_layout(
                title=f"Slice Placement Animation - Slice {i} ({slice_obj.slice_type.name}) - Energy Usage: {topology_consumption}",
                showlegend=True,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="closest",
            )
            if not complete_fig_name:
                fig.write_html(f"slice_{i}.html", auto_open=True)
                time.sleep(delay)
        if complete_fig_name:
            fig.write_html(f"{complete_fig_name}.html", auto_open=True)

    def _get_topology_consumption(self):
        total_energy = 0
        for node in self.topology.nodes():
            if self.topology.nodes[node]["cpu_usage"] > 0:
                total_energy += self.topology.nodes[node]["energy_base"]
            total_energy += (
                self.topology.nodes[node]["cpu_usage"]
                * self.topology.nodes[node]["energy_per_vcpu"]
            )
        return total_energy

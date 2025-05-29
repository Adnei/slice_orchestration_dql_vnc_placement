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
            energy_per_vcpu = self.topology.nodes[node].get("energy_per_vcpu", "N/A")

            # Short label for visibility
            node_trace["text"] += (f"{node} ({node_type})",)
            node_trace["x"] += (x,)
            node_trace["y"] += (y,)

            # Detailed hover text
            node_trace["hovertext"] += (
                f"<b>Node {node} ({node_type})</b><br>"
                f"CPU Usage: {cpu_usage}/{cpu_limit}<br>"
                f"Energy Base: {energy_base}<br>"
                f"Energy/VCPU: {energy_per_vcpu}",
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
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in self.topology.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]

            bw = self.topology.edges[edge].get("bandwidth", "N/A")
            delay = self.topology.edges[edge].get("delay", "N/A")

            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_text += [
                f"<b>Edge: {edge[0]} ↔ {edge[1]}</b><br>Bandwidth: {bw}<br>Delay: {delay}",
                f"<b>Edge: {edge[0]} ↔ {edge[1]}</b><br>Bandwidth: {bw}<br>Delay: {delay}",
                None,
            ]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.5, color="#888"),
            hoverinfo="text",
            text=edge_text,
            mode="lines",
        )
        return edge_trace

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

    def animate_slice_building(self, slices, delay=1.5):
        for i, slice_obj in enumerate(slices):
            fig = go.Figure()
            fig.add_trace(self._get_edge_trace())
            fig.add_trace(self._get_node_trace())

            for j in range(i + 1):
                if slices[j].path:
                    trace = self._get_slice_trace(slices[j], slices[j].slice_id)
                    fig.add_trace(trace)

            fig.update_layout(
                title=f"Slice Placement Animation - Slice {i} ({slice_obj.slice_type.name})",
                showlegend=True,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="closest",
            )

            fig.write_html(f"slice_{i}.html", auto_open=True)
            time.sleep(delay)

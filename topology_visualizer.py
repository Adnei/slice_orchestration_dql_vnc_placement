import plotly.graph_objects as go
import networkx as nx
import random
import time

# from IPython.display import display, clear_output


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
            node_trace["x"] += (x,)
            node_trace["y"] += (y,)
            node_type = self.topology.nodes[node]["type"]
            label = f"{node} ({node_type})"
            node_trace["text"] += (label,)
            color = {
                "RAN": "blue",
                "Edge": "orange",
                "Transport": "green",
                "Core": "red",
            }.get(node_type, "gray")
            node_trace["marker"]["color"] += (color,)

        return node_trace

    def _get_edge_trace(self):
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=1, color="#888"), hoverinfo="none", mode="lines"
        )
        for edge in self.topology.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_trace["x"] += (x0, x1, None)
            edge_trace["y"] += (y0, y1, None)
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
        )
        return trace

    def animate_slice_building(self, slices, delay=1.5):
        """
        Animates each slice being placed one at a time.
        Use in notebooks or scripts with IPython.display.
        """
        for i, slice_obj in enumerate(slices):
            fig = go.Figure()
            fig.add_trace(self._get_edge_trace())
            fig.add_trace(self._get_node_trace())

            # Add already completed slices up to now
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

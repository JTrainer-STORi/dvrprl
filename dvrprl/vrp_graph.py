import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class VRPGraph:
    graph: nx.Graph = nx.Graph()

    def __init__(self, num_nodes: int, num_depots: int, rng: np.random.Generator) -> None:
        """
        Creates a fully connected graph with node_num nodes
        and depot num depots. Coordinates of each node
        and the depot nodes will be sampled uniformly in the unit square.

        Args:
            num_nodes: Number of nodes in the graph.
            num_depots: Number of depots in the graph.
        """
        self.num_nodes = num_nodes
        self.num_depots = num_depots
        self.default_rng = rng

        # offset for demand node labels
        self.offset = offset = np.array([0, 0.065])

        # generate graph and set node positions
        self.graph = nx.complete_graph(num_nodes)
        node_position = {
            i: coordinates for i, coordinates in enumerate(rng.random((num_nodes, 2)))
        }
        nx.set_node_attributes(self.graph, node_position, "coordinates")

        # sample depots and set attributes
        self.depots = rng.choice(num_nodes, size=num_depots, replace=False)
        one_hot = np.zeros(num_nodes)
        one_hot[self.depots] = 1
        one_hot_dict = {i: depot for i, depot in enumerate(one_hot)}
        nx.set_node_attributes(self.graph, one_hot_dict, "depot")

        self.set_default_node_attributes()

    def set_default_node_attributes(self) -> None:
        """
        Sets the default colors of the nodes
        as attributes. Nodes are black except
        depots which are colored in red.

        Edges are initially marked as unvisited.
        """
        nx.set_edge_attributes(self.graph, False, "visited")
        nx.set_node_attributes(self.graph, "black", "node_color")
        for node in self.depots:
            self.graph.nodes[node]["node_color"] = "red"

    def draw(self) -> None:
        """
        Draws the graph as a matplotlib plot.
        Depots are colored in red. Edges that have been
        traveresed are colored in red. Nodes that have
        been visited are colored in grey.
        """

        plt.clf()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        # draw nodes according to color and position attribute
        pos = nx.get_node_attributes(self.graph, "coordinates")
        node_colors = nx.get_node_attributes(self.graph, "node_color").values()
        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_colors, ax=ax, node_size=100
        )

        # draw edges that were visited
        edges = [x for x in self.graph.edges(data=True) if x[2]["visited"]]
        nx.draw_networkx_edges(
            self.graph,
            pos,
            alpha=0.5,
            edgelist=edges,
            edge_color="red",
            ax=ax,
            width=1.5,
        )

        # draw demand above the node
        if self.plot_demand:
            demand_label_pos = {k: (v + self.offset) for k, v in pos.items()}
            node_demand = nx.get_node_attributes(self.graph, "demand")
            node_demand = {k: np.round(v, 2)[0] for k, v in node_demand.items()}
            nx.draw_networkx_labels(
                self.graph, demand_label_pos, labels=node_demand, ax=ax
            )

        plt.show()

    def visit_edge(self, source_node: int, target_node: int) -> None:
        """
        Sets the edge color to red and target node colour to grey.

        Args:
            source_node: Source node id of the edge
            target_node: Target node id of the edge
        """

        # don't draw self loops
        if source_node == target_node:
            return

        self.graph.edges[source_node, target_node]["visited"] = True

        # set node color to grey if it has been visited
        if self.graph.nodes[target_node]["node_color"] != "red":
            self.graph.nodes[target_node]["node_color"] = "grey"

    @property
    def demand(self) -> np.ndarray:
        positions = nx.get_node_attributes(self.graph, "demand").values()
        return np.asarray(list(positions))

    @property
    def edges(self):
        return self.graph.edges.data()

    @property
    def nodes(self):
        return self.graph.nodes.data()

    @property
    def node_positions(self) -> np.ndarray:
        """
        Returns the coordinates of each node as
        an ndarray of shape (num_nodes, 2) sorted
        by the node index.
        """

        positions = nx.get_node_attributes(self.graph, "coordinates").values()
        return np.asarray(list(positions))

    def get_distance(self, node1_idx: int, node2_idx: int) -> float:
        """
        Calculates the euclidian  distance between two nodes.

        Args:
            node1_idx: Index of the first node.
            node2_idx: Index of the second node.
        Returns:
            Euclidian distance between the two nodes.
        """

        node_one_pos = self.graph.nodes[node1_idx]["coordinates"]
        node_two_pos = self.graph.nodes[node2_idx]["coordinates"]

        return np.linalg.norm(node_one_pos - node_two_pos)

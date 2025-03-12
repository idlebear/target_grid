"""
graph-animation.py

Animate an arbitrary graph for testing, and solving instances of Stochatic
Shortest Paths
"""

import networkx as nx
import numpy as np

from .window import Colours

# set the display width for numpy to wide
np.set_printoptions(linewidth=np.inf)


class Graph:
    def __init__(
        self,
        edge_probability=0.5,
        seed=None,
        color_map="spring",
        method="random",
        **kwargs,
    ):

        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

        self.target_node = 0
        self.edge_probability = edge_probability
        self.color_map = color_map

        self.objects = {}

    def linear_index(self, index):
        """
        Convert the 2D index to a linear index, where the index is a tuple (x,y)
        """
        if self.grid_data is None or type(index) is int:
            return index
        return index[1] * self.cols + index[0]

    def grid_index(self, index):
        """
        Convert the linear index to a 2D index, where the index is a tuple (x,y)
        """
        if self.grid_data is None or type(index) is tuple:
            return index
        return (index % self.cols, index // self.cols)

    def get_connectivity_matrix(self):
        """
        Get the connectivity matrix of the graph.
        This function generates a connectivity matrix for the graph, where element
        (i, j) is 1 if there is an edge between nodes i and j, and 0 otherwise.
        Returns:
        numpy.ndarray: A 2D array representing the connectivity matrix of the graph.
        """

        matrix = np.zeros((self.n, self.n))

        for i, j in self.G.edges:
            matrix[self.linear_index(i), self.linear_index(j)] = 1
            matrix[self.linear_index(j), self.linear_index(i)] = 1

        return matrix

    def get_distance(self, start, goal):
        """
        Get the distance between two nodes in the graph.
        This function calculates the shortest path distance between two nodes in the
        graph using NetworkX's shortest_path_length function.
        Parameters:
        start (int): The starting node index.
        goal (int): The goal node index.
        Returns:
        int: The shortest path distance between the two nodes.
        """
        return nx.shortest_path_length(self.G, source=start, target=goal)

    def create_markov_transition_matrix(self):
        """
        Create a Markov transition matrix from a given graph.
        This function generates a transition matrix for a Markov chain based on the
        structure of the input graph. Each node in the graph represents a state, and
        the edges represent possible transitions between states. The transition
        probabilities are assigned randomly and normalized to ensure they sum to 1
        for each state.
        Parameters:
        graph (networkx.Graph): The input graph where nodes represent states and
                                edges represent possible transitions.
        Returns:
        numpy.ndarray: A 2D array representing the Markov transition matrix, where
                       element (i, j) is the probability of transitioning from state
                       i to state j.
        """

        matrix = np.zeros((self.n, self.n))

        for node in self.G.nodes:
            neighbors = list(self.G.neighbors(node)) + [node]
            if neighbors:
                probabilities = self.rng.random(len(neighbors))
                probabilities /= probabilities.sum()
                node_index = self.linear_index(node)
                for neighbor, p in zip(neighbors, probabilities):
                    neighbor_index = self.linear_index(neighbor)
                    matrix[node_index, neighbor_index] = p

        return matrix

    @staticmethod
    def prep_graph(G, shape=None):
        # Initialize the position of nodes for consistent layout
        if shape is not None:
            # labels = dict(((i, j), i + j * shape[1]) for i, j in G.nodes())
            labels = dict((n, n) for n in G.nodes())
            pos = dict((n, n) for n in G.nodes())  # Dictionary of all positions
        else:
            labels = dict((n, n) for n in G.nodes())
            pos = nx.spring_layout(G)

        nx.set_node_attributes(G, pos, "pos")
        nx.set_node_attributes(G, labels, "labels")

        # Set all weights to 1
        for edge in G.edges:
            G.edges[edge]["weight"] = 1

    @staticmethod
    def create_undirected_graph(n, edge_probability=0.5, generator=None, **kwargs):
        """
        Create a random graph with a specified number of nodes and edge probability.
        This function generates a complete graph with `n` nodes and then removes edges
        randomly based on the given `edge_probability` while ensuring the graph remains
        connected.
        Parameters:
        n (int): The number of nodes in the graph.
        edge_probability (float, optional): The probability of retaining an edge in
            the graph. Default is 0.5.
        Returns:
        networkx.Graph: A randomly generated graph with `n` nodes.
        """

        G = nx.complete_graph(n)
        Graph.prep_graph(G)

        if generator is None:
            shuffle = np.random.shuffle
            remove_edge = np.random.random
        else:
            shuffle = generator.shuffle
            remove_edge = generator.random

        # Remove edges randomly while ensuring the graph remains connected
        edges = list(G.edges)
        shuffle(edges)
        for edge in edges:
            if remove_edge() > edge_probability:
                G.remove_edge(*edge)
                if not nx.is_connected(G):
                    G.add_edge(
                        *edge
                    )  # Re-add the edge if the graph becomes disconnected

        return G

    @staticmethod
    def create_grid_graph(
        grid: np.array, edge_probability=0.5, generator=None, **kwargs
    ):
        """
        Create a Euclidean graph with a specified grid size and edge probability.
        This function generates a grid graph with the specified `grid` size and then
        removes edges randomly based on the given `edge_probability` while ensuring the
        graph remains connected.
        Parameters:
        grid (tuple): The grid size of the graph as a tuple (rows, cols).
        edge_probability (float, optional): The probability of retaining an edge
                                            in the graph. Default is 0.5.
        Returns:
        networkx.Graph: A randomly generated Euclidean graph with the specified
                        grid size.
        """

        rows, cols = grid.shape
        G = nx.grid_2d_graph(cols, rows)
        Graph.prep_graph(G, shape=(rows, cols))

        if "connectivity" in kwargs and kwargs["connectivity"] == "euclidean":
            # Add diagonal connections
            for x in range(cols - 1):
                for y in range(rows - 1):
                    G.add_edge((x, y), (x + 1, y + 1), weight=1)
                    G.add_edge((x + 1, y), (x, y + 1), weight=1)

        # add self loops
        for node in G.nodes:
            G.add_edge(node, node, weight=1)

        if generator is None:
            shuffle = np.random.shuffle
            remove_edge = np.random.random
        else:
            shuffle = generator.shuffle
            remove_edge = generator.random

        # Remove edges randomly while ensuring the graph remains connected
        if edge_probability < 1:
            edges = list(G.edges)
            shuffle(edges)
            for edge in edges:
                if remove_edge() > edge_probability:
                    G.remove_edge(*edge)
                    if not nx.is_connected(G):
                        G.add_edge(*edge)

        # disconnect any nodes where the cell is occupied in the grid
        for node in G.nodes:
            x, y = node
            if grid[y, x]:
                # disconnect all edges from this node
                edges = list(G.edges(node))
                for edge in edges:
                    G.remove_edge(*edge)

        return G

    def validate_node(self, node, next_node):
        """
        Check if the next node is a valid node in the graph
        """
        neighbours = list(self.G.neighbors(node))
        if next_node in neighbours:
            return next_node
        return node


class GridGraph(Graph):
    def __init__(
        self,
        edge_probability=0.5,
        seed=None,
        color_map="spring",
        method="euclidean",
        **kwargs,
    ):

        super().__init__(edge_probability, seed, color_map, method, **kwargs)

        self.target_node = 0
        self.edge_probability = edge_probability
        self.color_map = color_map

        self.grid_data = kwargs.get("grid_data", None)
        if self.grid_data is None:
            raise ValueError("grid must be specified for euclidean graph")
        self.rows, self.cols = self.grid_data.shape
        self.G = Graph.create_grid_graph(
            self.grid_data, self.edge_probability, connectivity=method
        )
        self.n = self.rows * self.cols
        self.node_states = np.zeros(self.n)

        self.node_positions = nx.get_node_attributes(self.G, "pos")
        self.min_position = np.min(list(self.node_positions.values()), axis=0)
        self.max_position = np.max(list(self.node_positions.values()), axis=0)

    def draw(self, window, current_visibility):
        """
        Draw the graph on the window object (using Window.py)
        """

        for node in self.G.nodes:
            x, y = node[0] + 0.5, node[1] + 0.5

            # set the hue based on the node visibility
            colour = np.array(Colours.light_blue) * current_visibility[
                node[1], node[0]
            ] + (1 - current_visibility[node[1], node[0]]) * np.array(
                Colours.light_grey
            )
            window.draw_rect(
                center=(x, y),
                height=1,
                width=1,
                colour=colour,
                border_colour=Colours.black,
                border_width=1,
            )

    def get_distances(self, node):
        """Use bellman-ford to get the shortest path from the node to all other nodes"""
        return nx.single_source_bellman_ford_path_length(self.G, node)


class UndirectedGraph(Graph):
    def __init__(
        self,
        edge_probability=0.5,
        seed=None,
        color_map="spring",
        method="random",
        **kwargs,
    ):
        super().__init__(edge_probability, seed, color_map, method, **kwargs)

        self.n = kwargs.get("n", 5)
        if self.n is None:
            raise ValueError("n must be specified for random graph")
        self.G = Graph.create_undirected_graph(
            self.n, self.edge_probability, generator=self.rng
        )
        self.grid_data = None

        self.node_positions = nx.spring_layout(self.G, seed=747, scale=1, center=(0, 0))
        self.min_position = np.min(list(self.node_positions.values()), axis=0)
        self.max_position = np.max(list(self.node_positions.values()), axis=0)

    def draw(self, window):
        """
        Draw the graph on the window object (using Window.py)
        """

        for node in self.G.nodes:
            pos = self.G.nodes[node]["pos"]
            label = self.G.nodes[node]["labels"]
            window.draw_circle(center=pos, colour=Colours.light_blue, radius=0.1)
            window.draw_text(pos, label)

        super().draw(window)

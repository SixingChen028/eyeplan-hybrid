import numpy as np
import matplotlib.pyplot as plt
import random


class Graph:
    """
    A graph class.
    """

    def __init__(
            self,
            num_nodes = 11,
            point_set = np.array([-8, -4, -2, -1, 1, 2, 4, 8])
        ):
        """
        Initialize the graph.
        """
        
        # initialize parameters
        self.num_nodes = num_nodes # total number of nods
        self.point_set = point_set # point set

        # initialize graph structures
        # only 4 topological unique tree structures
        self.possible_child_dicts = [
            {
                0: [1, 2],
                2: [3, 4],
                4: [5, 6],
                6: [7, 8],
                8: [9, 10]
            },
            {
                0: [1, 2],
                1: [3, 4],
                2: [5, 6],
                6: [7, 8],
                8: [9, 10],
            },
            {
                0: [1, 2],
                1: [3, 4],
                2: [5, 6],
                5: [7, 8],
                6: [9, 10],
            },
            {
                0: [1, 2],
                1: [3, 4],
                2: [5, 6],
                4: [7, 8],
                6: [9, 10],
            }
        ]
        

    def reset(
            self,
            shuffle_nodes = True,
        ):
        """
        Reset the graph.
        """

        # sample a base graph
        self.base_graph = np.random.choice(self.possible_child_dicts)

        # initialize and shuffle nodes
        if shuffle_nodes:
            # shuffle nodes
            ordered_nodes = np.arange(self.num_nodes)
            shuffled_nodes = np.random.permutation(ordered_nodes)
            mapping = dict(zip(ordered_nodes, shuffled_nodes))

            # initialize nodes
            self.nodes = shuffled_nodes

            # initialize child_dicts
            self.child_dict = {mapping[parent]: [mapping[child] for child in children] for parent, children in self.base_graph.items()}
        
        else:
            # initialize nodes
            self.nodes = np.arange(self.num_nodes)

            # initialize child_dicts
            self.child_dict = self.base_graph

        # initialize parent node dict
        self.parent_dict = {v: k for k, values in self.child_dict.items() for v in values}

        # get leaf nodes
        child_keys = np.array(list(self.child_dict.keys()))
        parent_keys = np.array(list(self.parent_dict.keys()))
        self.leaf_nodes = parent_keys[~np.isin(parent_keys, child_keys)]

        # get max depth
        self.max_depth = int((self.num_nodes - 1) / 2)

        # get root node and non-leaf nodes
        self.root_node = self.nodes[0]
        self.non_leaf_nodes = np.array(list(self.child_dict.keys()))
        
        # get node counts
        self.num_leaf = len(self.leaf_nodes)
        self.num_non_leaf = len(self.non_leaf_nodes)

        # initialize points
        self.points = np.random.choice(self.point_set, size = self.num_nodes, replace = True)
        self.points[self.root_node] = 0.

        # compute cumulative points
        self.cum_points = self.get_cum_points()

        # get point dict
        self.point_dict = {}
        for node in self.nodes:
            self.point_dict[node] = self.points[node]
            

    def successors(self, node):
        """
        Find successor nodes of a given node.
        """

        # no child if node not in tree or node is leaf
        if not self.in_tree(node) or node in self.leaf_nodes:
            return [None, None]
        else:
            return self.child_dict[node]


    def predecessors(self, node):
        """
        Find predecessor nodes of a given node.
        """

        # no parent if node not in tree or node is root
        if not self.in_tree(node) or node == self.root_node:
            return None
        else:
            return self.parent_dict[node]
    
    
    def in_tree(self, node):
        """
        Check if a node is in the tree.
        """

        in_tree = (node in self.parent_dict or node == self.root_node)

        return in_tree


    def get_cum_points(self):
        """
        Get cumulative points.
        """

        cum_points = np.zeros((self.num_nodes,))

        # helper depth-first-search function
        def _dfs(node, cum):
            # update the current sum
            cum += self.points[node]
            # store the current sum
            cum_points[node] = cum
            
            # continue dfs
            if node in self.child_dict.keys():
                for child in self.child_dict[node]:
                    _dfs(child, cum)
        
        # execute dfs
        _dfs(self.root_node, 0)

        return cum_points
    

    def get_depths(self):
        """
        Get depths.
        """

        depths = np.zeros((self.num_nodes,), dtype = int) 

        # helper depth-first-search function
        def _dfs(node, depth):
            depths[node] = depth
            if node in self.child_dict.keys():
                for child in self.child_dict[node]:
                    _dfs(child, depth + 1)
        
        # execute dfs
        _dfs(self.root_node, 0)
        
        return depths
    

    def get_adj_list(self):
        """
        Get adjacency list.
        """

        self.adj_list = [[] for _ in range(self.num_nodes)]
        for node, children in self.child_dict.items():
            self.adj_list[node] = children

        return self.adj_list


    def get_adj_matrix(self):
        """
        Get adjacency matrix.
        """

        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for node, children in self.child_dict.items():
            for child in children:
                self.adj_matrix[node, child] = 1
        
        return self.adj_matrix
    




if __name__ == '__main__':
    # testing
    import networkx as nx

    graph = Graph(
        num_nodes = 11,
        point_set = np.array([-8, -4, -2, -1, 1, 2, 4, 8])
    )

    graph.reset(
        shuffle_nodes = True
    )

    print('child dict:', graph.child_dict)
    print('parent dict:', graph.parent_dict)
    print('points:', graph.points)
    print('root node:', graph.root_node)
    print('leaf nodes:', graph.leaf_nodes)
    print('non-leaf nodes:', graph.non_leaf_nodes)
    print('cumulative points:', graph.cum_points)
    print('depths:', graph.get_depths())
    print('adjacency list:', graph.get_adj_list())
    print('adjacency matrix:', graph.get_adj_matrix())

    G = nx.from_numpy_array(graph.adj_matrix)
    plt.figure(figsize = (4, 4))
    nx.draw_circular(G, with_labels = True)
    plt.show()
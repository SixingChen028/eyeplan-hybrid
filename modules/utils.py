import numpy as np


def relationship(child_dict, node1, node2):
    """
    A function for testing the relationship of two nodes in a tree.
    """

    if node1 == node2:
        return 'self'
    elif node1 in child_dict.keys() and node2 in child_dict[node1]:
        return 'child'
    elif node2 in child_dict.keys() and node1 in child_dict[node2]:
        return 'parent'
    else:
        for parent, children in child_dict.items():
            if node1 in children and node2 in children:
                return 'sibling'
    return 'others'


def distance(child_dict, start, end, visited = None):
    """
    A function for measureing the distance between two nodes in a tree.
    """

    if visited is None:
        visited = set()

    def build_adjacency_list(tree):
        adjacency_list = {}
        for parent, children in tree.items():
            for child in children:
                adjacency_list.setdefault(parent, []).append(child)
                adjacency_list.setdefault(child, []).append(parent) # for undirected tree
        return adjacency_list

    def dfs(adjacency_list, current, target, visited):
        if current == target:
            return 0
        visited.add(current)
        for neighbor in adjacency_list[current]:
            if neighbor not in visited:
                distance = dfs(adjacency_list, neighbor, target, visited)
                if distance >= 0:
                    return distance + 1
        return -1  # indicating target node not found from current node

    adjacency_list = build_adjacency_list(child_dict)

    return dfs(adjacency_list, start, end, visited)


def get_q_values(child_dict, points, num_node = 11, gamma = 1):
    """
    Get Q values (in points).
    """

    q_values = np.zeros((num_node,))

    # helper reverse depth-first-search function
    def _rdfs(node):
        if node not in child_dict.keys():
            # if the node is a leaf node, return point as q value
            q_values[node] = points[node]
            return q_values[node]
        
        # otherwise, compute the discounted maximum q value of its children
        max_child_q_value = max(_rdfs(child) for child in child_dict[node])

        # set the q value
        q_values[node] = points[node] + gamma * max_child_q_value

        return q_values[node]
    
    # execute dfs from the root node
    _rdfs(list(child_dict.keys())[0])

    return q_values


def child_dict_to_adj_list(child_dict, num_node = 11):
    """
    Transfer child dict to adjacency list.
    """

    adj_list = [[] for _ in range(num_node)]

    for node, children in child_dict.items():
        adj_list[node] = children
    
    return adj_list


def adj_list_to_child_dict(adj_list):
    """
    Transfer adjacency list to child dict.
    """

    child_dict = {}
    
    for node, children in enumerate(adj_list):
        if children:
            child_dict[node] = children
    
    return child_dict


def merge(lst):
    """
    Merge identical adjacent elements in a list.
    """

    # check if the list is empty
    if not lst:
        return []

    # start with the first element
    merged_list = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            merged_list.append(lst[i])

    return merged_list


def count_intervals(node_seq, errors_seq):
    """
    Count refixation intervals.
    """
    
    # initialize result
    result = {}
    length = len(node_seq)
    
    # iterate through unique elements
    for node in np.unique(node_seq):

        # get count
        count = np.sum(node_seq == node)
        
        # if refixate
        if count > 1:
            
            result[node] = {
                'count': count,
                'intervals': [],
                'indices': []
            }
            
            # get indices
            indices = np.where(node_seq == node)[0]
            result[node]['indices'] = list(indices)
            
            # get intervals
            intervals = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
            result[node]['intervals'] = intervals

            # get slices
            slices = []
            for i in range(len(indices) - 1):
                start = indices[i]
                end = indices[i + 1] + 2 ##############

                # break if exceeding episode length
                if end >= length:
                    break
                
                slices.append(errors_seq[start:end, node]) # include the refixate timestep

            result[node]['slices'] = slices
    
    return result


def get_node_depths(child_dict, root_node):
    """
    Get node depths.
    """
    
    depths = {}

    def traverse(node, depth):
        # set the depth of the current node
        depths[node] = depth
        # recursively traverse the children if they exist
        if node in child_dict:
            for child in child_dict[node]:
                traverse(child, depth + 1)

    # start traversal from the root node with depth 0
    traverse(root_node, 0)
    return depths


def get_cum_points(child_dict, root_node, points):
    """
    Get cumulative points.
    """

    cum_points = np.zeros(len(points))

    # helper depth-first-search function
    def _dfs(node, cum):
        # update the current sum
        cum += points[node]
        # store the current sum
        cum_points[node] = cum
        
        # continue dfs
        if node in child_dict.keys():
            for child in child_dict[node]:
                _dfs(child, cum)
    
    # execute dfs
    _dfs(root_node, 0)

    return cum_points


def normalize_logits(child_dict, logits_seq, num_node = 11):
    """
    Normalize action logits between sibling nodes.
    """

    # initialize normalized logits sequence as a copy of the original logits_seq
    normalized_logits_seq = np.copy(logits_seq)
    
    # normalize logits between sibling states
    for parent, children in child_dict.items():
        child1, child2 = children
        normalized_logits_seq[:, child1] = logits_seq[:, child1] - logits_seq[:, child2]
        normalized_logits_seq[:, child2] = logits_seq[:, child2] - logits_seq[:, child1]

        normalized_logits_seq[:, num_node + child1] = logits_seq[:, num_node + child1] - logits_seq[:, num_node + child2]
        normalized_logits_seq[:, num_node + child2] = logits_seq[:, num_node + child2] - logits_seq[:, num_node + child1]
    
    return normalized_logits_seq
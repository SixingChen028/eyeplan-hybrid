import numpy as np
import random
import pickle
import torch
import warnings
warnings.filterwarnings('ignore')

from .utils import *


# def simulate(
#         net,
#         env,
#         num_trials,
#         greedy = False,
#     ):
#     """
#     Simulate.
#     """

#     # reset environment
#     env.reset()

#     # reset data
#     data = {
#         'child_dicts': [],
#         'parent_dicts':[],
#         'root_nodes': [],
#         'leaf_nodes': [],
#         'depths': [],
#         'points': [],
#         'cum_points': [],
#         'action_seqs': [],
#         'choice_seqs': [],
#     }

#     # iterate through trials
#     for _ in range(num_trials):

#         # initialize trial recordings
#         action_seq_ep = []

#         # initialize a trial
#         done = False

#         # reset environment
#         obs, info = env.reset()
#         obs = torch.Tensor(obs).unsqueeze(dim = 0) # (1, feature_dim)
#         action_mask = torch.tensor(info['mask']) # (action_dim,)

#         with torch.no_grad():
#             # iterate through a trial
#             while not done:

#                 # step the net
#                 action, policy, log_prob, entropy, value = net(
#                     obs, action_mask
#                 )
#                 if greedy:
#                     action = torch.argmax(policy)

#                 # step the env
#                 obs, reward, done, truncated, info = env.step(action.item())
#                 obs = torch.Tensor(obs).unsqueeze(dim = 0) # (1, feature_dim)
#                 action_mask = torch.tensor(info['mask']) # (action_dim,)

#                 # record results for the timestep
#                 action_seq_ep.append(int(action))

#             # record results for the trial
#             data['child_dicts'].append(env.graph.child_dict)
#             data['parent_dicts'].append(env.graph.parent_dict)
#             data['root_nodes'].append(int(env.graph.root_node))
#             data['leaf_nodes'].append(list(env.graph.leaf_nodes))
#             data['points'].append(list(env.graph.points))
#             data['cum_points'].append(list(env.graph.cum_points))
#             data['depths'].append(list(env.graph.get_depths()))
#             data['action_seqs'].append(action_seq_ep)
#             data['choice_seqs'].append(env.chosen_path)
    
#     return data



def simulate(
        net,
        env,
        num_trials,
        greedy = False,
    ):
    """
    Simulate, and remap each trial's 11 realized node IDs to {0..10}
    so downstream analysis written for 11 nodes works unchanged.
    """

    def collect_present_nodes(child_dict, root):
        """Return the set of nodes that actually appear in this trial's tree."""
        present = set([root])
        stack = [root]
        while stack:
            n = stack.pop()
            for c in child_dict.get(n, []):
                if c not in present:
                    present.add(c)
                    stack.append(c)
        return present

    def remap_child_dict(child_dict, node_map):
        return {node_map[p]: [node_map[c] for c in ch]
                for p, ch in child_dict.items()
                if p in node_map}

    def remap_parent_dict(parent_dict, node_map):
        return {node_map[ch]: node_map[p]
                for ch, p in parent_dict.items()
                if ch in node_map and p in node_map}

    def remap_vector(vec63, present_nodes, node_map, size=11):
        """
        vec63: list/np.array length 63 (or >= max node id + 1)
        returns list length 11 where entry[new_id] = vec63[old_id]
        """
        out = [0.0] * size
        for old in present_nodes:
            out[node_map[old]] = float(vec63[old])
        return out

    # reset environment once (not strictly needed, but keeps behavior same as your version)
    env.reset()

    data = {
        'child_dicts': [],
        'parent_dicts': [],
        'root_nodes': [],
        'leaf_nodes': [],
        'depths': [],
        'points': [],
        'cum_points': [],
        'action_seqs': [],
        'choice_seqs': [],
        # optional: keep the mapping if you ever want to invert it later
        'node_maps': [],  # maps old_id -> new_id
    }

    for _ in range(num_trials):
        action_seq_ep = []
        done = False

        obs, info = env.reset()
        obs = torch.Tensor(obs).unsqueeze(dim=0)  # (1, feature_dim)
        action_mask = torch.tensor(info['mask'])  # (action_dim,)

        with torch.no_grad():
            while not done:
                action, policy, log_prob, entropy, value = net(obs, action_mask)
                if greedy:
                    action = torch.argmax(policy)

                obs, reward, done, truncated, info = env.step(action.item())
                obs = torch.Tensor(obs).unsqueeze(dim=0)
                action_mask = torch.tensor(info['mask'])

                action_seq_ep.append(int(action))

        # ----------------------------
        # Build a random mapping: trial nodes -> {0..10}
        # ----------------------------
        child_dict_full = env.graph.child_dict
        parent_dict_full = env.graph.parent_dict
        root_full = int(env.graph.root_node)

        present_nodes = collect_present_nodes(child_dict_full, root_full)
        present_nodes = sorted(list(present_nodes))
        if len(present_nodes) != 11:
            raise ValueError(f"Expected 11 present nodes, got {len(present_nodes)}")

        perm = np.random.permutation(11).tolist()
        node_map = {old: new for old, new in zip(present_nodes, perm)}  # old_id -> 0..10

        # Remap structures
        child_dict_11 = remap_child_dict(child_dict_full, node_map)
        parent_dict_11 = remap_parent_dict(parent_dict_full, node_map)
        root_11 = node_map[root_full]

        leaf_nodes_full = list(env.graph.leaf_nodes)
        leaf_nodes_11 = [node_map[n] for n in leaf_nodes_full if n in node_map]

        # Remap per-node arrays (63 -> 11)
        points_11 = remap_vector(env.graph.points, present_nodes, node_map, size=11)
        cum_points_11 = remap_vector(env.graph.cum_points, present_nodes, node_map, size=11)
        depths_full = env.graph.get_depths()
        depths_11 = remap_vector(depths_full, present_nodes, node_map, size=11)
        depths_11 = [int(_) for _ in depths_11]

        # Remap action sequence:
        # - fixations: old node id -> new id in 0..10
        # - move action: env.action_space.n - 1 (i.e., 63) -> 11
        move_action_full = env.action_space.n - 1
        move_action_11 = 11

        action_seq_11 = []
        for a in action_seq_ep:
            if a == move_action_full:
                action_seq_11.append(move_action_11)
            else:
                if a not in node_map:
                    raise ValueError(f"Action {a} is not in trial node_map. Present nodes: {present_nodes}")
                action_seq_11.append(node_map[a])

        # Remap chosen path (list of old node ids) -> 0..10
        choice_seq_full = list(env.chosen_path)
        choice_seq_11 = [node_map[n] for n in choice_seq_full if n in node_map]

        # Save remapped trial
        data['child_dicts'].append(child_dict_11)
        data['parent_dicts'].append(parent_dict_11)
        data['root_nodes'].append(int(root_11))
        data['leaf_nodes'].append(leaf_nodes_11)
        data['points'].append(points_11)
        data['cum_points'].append(cum_points_11)
        data['depths'].append(depths_11)
        data['action_seqs'].append(action_seq_11)
        data['choice_seqs'].append(choice_seq_11)
        data['node_maps'].append(node_map)

    return data


def preprocess(data, args, merge_fixations = False):
    """
    Preprocess data.
    """

    num_trials = len(data['action_seqs'])

    # add variables
    data['lengths'] = []
    data['fixation_seqs'] = []
    data['decision_seqs'] = []

    for i in range(num_trials):
        
        action_seq_ep, choice_seq_ep = pull(data, i, 'action_seqs', 'choice_seqs')

        fixation_seq_ep = [action for action in action_seq_ep if action < args.num_nodes]
        decision_seq_ep = [action for action in choice_seq_ep]

        data['lengths'].append(len(action_seq_ep))
        if merge_fixations:
            data['fixation_seqs'].append(merge(fixation_seq_ep))
        else:
            data['fixation_seqs'].append(fixation_seq_ep)
        data['decision_seqs'].append(decision_seq_ep)

    return data


def pull(data, index, *keys):
    """
    Pull data according to keys.
    """
    return [data[key][index] for key in keys]


def save_data(data, path):
    """
    Save data.
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_data(path):
    """
    Load data.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


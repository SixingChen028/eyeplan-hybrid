import numpy as np
import random
import pickle
import torch
import warnings
warnings.filterwarnings('ignore')

from .utils import *


def simulate(
        net,
        env,
        num_trials,
        greedy = False,
    ):
    """
    Simulate.
    """

    # reset environment
    env.reset()

    # reset data
    data = {
        'child_dicts': [],
        'parent_dicts':[],
        'root_nodes': [],
        'leaf_nodes': [],
        'depths': [],
        'points': [],
        'cum_points': [],
        'action_seqs': [],
        'choice_seqs': [],
    }

    # iterate through trials
    for _ in range(num_trials):

        # initialize trial recordings
        action_seq_ep = []

        # initialize a trial
        done = False

        # reset environment
        obs, info = env.reset()
        obs = torch.Tensor(obs).unsqueeze(dim = 0) # (1, feature_dim)
        action_mask = torch.tensor(info['mask']) # (action_dim,)

        with torch.no_grad():
            # iterate through a trial
            while not done:

                # step the net
                action, policy, log_prob, entropy, value = net(
                    obs, action_mask
                )
                if greedy:
                    action = torch.argmax(policy)

                # step the env
                obs, reward, done, truncated, info = env.step(action.item())
                obs = torch.Tensor(obs).unsqueeze(dim = 0) # (1, feature_dim)
                action_mask = torch.tensor(info['mask']) # (action_dim,)

                # record results for the timestep
                action_seq_ep.append(int(action))

            # record results for the trial
            data['child_dicts'].append(env.graph.child_dict)
            data['parent_dicts'].append(env.graph.parent_dict)
            data['root_nodes'].append(int(env.graph.root_node))
            data['leaf_nodes'].append(list(env.graph.leaf_nodes))
            data['points'].append(list(env.graph.points))
            data['cum_points'].append(list(env.graph.cum_points))
            data['depths'].append(list(env.graph.get_depths()))
            data['action_seqs'].append(action_seq_ep)
            data['choice_seqs'].append(env.chosen_path)
    
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


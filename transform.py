import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from modules import *





"""
Set environment
"""

# set random seed
seed = 15
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# parse args
parser = ArgParser()
args = parser.args

# set experiment path
exp_path = os.path.join(args.path, f'exp_{args.learning_rate}_{args.jobid}')





"""
Load data
"""

# load data
with open(os.path.join(exp_path, 'data_simulation.p'), 'rb') as f:
    data_pickle = pickle.load(f)
print(data_pickle.keys())

# initialize new data
data_json = {
    'adj_lists': [],
    'starts': [],
    'rewards': [],
    'actions': [],
}





"""
Process data
"""

for i in range(len(data_pickle['child_dicts'])):
    length_ep = len(data_pickle['action_seqs'][i])

    if length_ep < args.t_max:

        # adj list
        child_dict_ep = data_pickle['child_dicts'][i]
        data_json['adj_lists'].append(child_dict_to_adj_list(child_dict_ep))

        # starts
        root_node_ep = data_pickle['root_nodes'][i]
        data_json['starts'].append(root_node_ep)

        # rewards
        points_ep = data_pickle['points'][i]
        data_json['rewards'].append(points_ep)
        
        # actions
        action_seq_ep = data_pickle['action_seqs'][i]
        choice_seq_ep = data_pickle['choice_seqs'][i]

        decision_seq_ep = [action + args.num_nodes for action in choice_seq_ep]

        if action_seq_ep[-1] == args.num_nodes:
            action_seq_ep[-1] = args.num_nodes * 2
        else:
            print('unfinished')
        action_seq_ep = action_seq_ep + decision_seq_ep
        data_json['actions'].append(action_seq_ep)





"""
custom json encoder
"""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.float64)):
            return obj.item()
        return super().default(obj)





"""
Save data
"""

# set ouput path
output_path = os.path.join(args.path, f'data_json')

# save data
with open(os.path.join(output_path, f'data_{args.learning_rate}_{args.jobid}.json'), 'w') as file:
    json.dump(data_json, file, cls = NumpyEncoder)
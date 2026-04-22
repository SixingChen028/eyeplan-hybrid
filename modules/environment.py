import numpy as np
import random

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete

from .graph import *
from .planner import *
# from graph import * # debugging use


class DecisionTreeEnv(gym.Env):
    """
    A decision tree environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            num_nodes = 15,
            beta_move = 4.0,
            eps_move = 0.02,
            learning_rate = 0.2,
            lamda_backup = 0.0,
            wm_decay = 0.8,
            t_max = 100,
            cost = 0.01,
            scale_factor = 1 / 8,
            shuffle_nodes = True,
            mask_fixation = True,
            seed = None,
        ):
        """
        Construct an environment.
        """

        self.num_nodes = num_nodes # number of nodes
        self.wm_decay = wm_decay # working memory decay
        self.t_max = t_max # max time steps per episode
        self.cost = cost # cost per action
        self.scale_factor = scale_factor # reward scale factor
        self.shuffle_nodes = shuffle_nodes # if shuffle nodes
        self.mask_fixation = mask_fixation # if use fixation mask

        self.graph = Graph(
            num_nodes = self.num_nodes,
            point_set = np.array([-8, -4, -2, -1, 1, 2, 4, 8])
        )

        self.planner = Planner(
            beta_move = beta_move,
            eps_move = eps_move,
            learning_rate = learning_rate,
            lamda_backup = lamda_backup,
        )

        # set random seed
        self.set_random_seed(seed)

        # initialize point set
        self.point_set = self.graph.point_set

        # initialize action space
        self.action_space = Discrete(self.num_nodes + 1)

        # initialize observation space
        observation_shape = (
            self.num_nodes + # fixation node (num_nodes,)
            1 + # point (1,)
            self.num_nodes * 3 + # parent and childs of fixation node (3 * num_nodes,)
            self.num_nodes + # root node
            self.num_nodes + # g values
            self.num_nodes + # q values
            self.num_nodes + # fixation counts
            1, # time
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = observation_shape,)


    def reset(self, seed = None, options = None):
        """
        Reset the environment.
        """

        # reset the trial
        self.init_trial()

        # get observation
        obs = self.get_obs()
    
        # get info
        info = {
            'mask': self.get_action_mask(),
        }

        return obs, info


    def step(self, action):
        """
        Step the environment.
        """

        action = int(action)

        self.time_elapsed += 1
        if self.time_elapsed == self.t_max:
            action = self.num_nodes
        done = False
        reward = -self.cost # initialize reward as cost

        # fixation
        if action < self.num_nodes:
            # debugging
            if action not in self.planner.parents.keys():
                raise ValueError('Fixated node not in the partial tree.')

            # fixate
            self.planner.look(action)

            # update fixated node
            self.fixation_node = action

            # update wm activation
            self.update_activation(action)

        # move
        elif action == self.num_nodes:
            # move
            cum_reward, chosen_path = self.planner.move()
            reward = cum_reward * self.scale_factor

            # record chosen path
            self.chosen_path = chosen_path

        # done
        if action == self.num_nodes or self.time_elapsed == self.t_max:
            done = True

        # get observation
        obs = self.get_obs()
    
        # get info
        info = {
            'mask': self.get_action_mask(),
        }

        return obs, reward, done, False, info
    
    
    def init_trial(self):
        """
        Initialize a trial.
        """

        # initialize time elapsed and stage
        self.time_elapsed = 0
        self.chosen_path = []

        # initialize the tree
        self.graph.reset(shuffle_nodes = self.shuffle_nodes)

        # initialize the planner
        self.planner.init_problem(
            children = self.graph.child_dict,
            rewards = self.graph.point_dict,
            root = self.graph.root_node
        )

        # initialize fixation node
        self.fixation_node = self.graph.root_node

        # initialize activation
        self.activation = np.zeros(self.num_nodes, dtype = np.float32)
        self.active_mask = np.zeros(self.num_nodes, dtype = bool)

        # make root and its children initially active
        self.update_activation(self.graph.root_node)


    def update_activation(self, node):
        """
        Update activation of all the nodes given the current fixated node.
            decay -> boost local neighborhood -> stochastic dropout
        """

        # decay
        self.activation *= self.wm_decay
        self.activation = np.clip(self.activation, 0.0, 1.0) # make sure between 0 and 1

        # boost fixated node
        self.activation[node] = 1.0

        # boost parent
        parent = self.graph.predecessors(node)
        if parent is not None:
            self.activation[parent] = 1.0

        # boost children
        child1, child2 = self.graph.successors(node)
        if child1 is not None:
            self.activation[child1] = 1.0
        if child2 is not None:
            self.activation[child2] = 1.0
        
        # boost root
        self.activation[self.graph.root_node] = 1.0

        # stochastic dropout
        keep = np.random.rand(self.num_nodes) < self.activation

        # *** zero out dropped nodes so they cannot reappear spontaneously ***
        self.activation[~keep] = 0.0

        # apply mask
        self.active_mask = keep


    def get_obs(self):
        """
        Get observation.
        """

        # get parent and child nodes
        fixation_parent_node = self.graph.predecessors(self.fixation_node)
        fixation_child_nodes = self.graph.successors(self.fixation_node)

        # get root node
        root_node = self.graph.root_node

        # wrap observation
        obs = np.hstack([
            self.one_hot_coding(num_classes = self.num_nodes, labels = self.fixation_node),
            self.graph.points[self.fixation_node],
            self.one_hot_coding(num_classes = self.num_nodes, labels = fixation_parent_node),
            self.one_hot_coding(num_classes = self.num_nodes, labels = fixation_child_nodes[0]),
            self.one_hot_coding(num_classes = self.num_nodes, labels = fixation_child_nodes[1]),
            self.one_hot_coding(num_classes = self.num_nodes, labels = root_node),
            self.get_path_values(),
            self.get_q_values(),
            self.get_num_visits(),
            self.time_elapsed,
        ])

        return obs
    

    def get_q_values(self):
        """
        Get q values.
        """

        q_values = np.zeros(self.num_nodes)

        for node, q_value in self.planner.q.items():
            q_values[node] = q_value
        
        return q_values


    def get_path_values(self):
        """
        Get path values.
        """

        path_values = np.zeros(self.num_nodes)

        for node, path_value in self.planner.g.items():
            path_values[node] = path_value
        
        return path_values
    

    def get_num_visits(self):
        """
        Get number of visits.
        """

        num_visits = np.zeros(self.num_nodes)

        for node, num in self.planner.n_visits.items():
            num_visits[node] = num
        
        return num_visits


    def get_action_mask(self):
        """
        Get action mask.

        Note:
            no batching is considered here. batching is implemented by vectorzation wrapper.
            if no batch training is used, add the batch dimension and transfer the mask to torch.tensor in trainer.
            if batch training is used, concatenate batches and transfer the mask to torch.tensor in trainer.
        """

        mask = np.zeros((self.action_space.n,), dtype = bool)

        # move is allowed
        mask[-1] = True

        # legal fixation nodes
        legal_mask = np.zeros(self.num_nodes, dtype = bool)
        for node in self.planner.parents.keys(): # root always inluded
            legal_mask[node] = True
        
        # unmasked if both legal and active
        gated_mask = legal_mask & self.active_mask

        # safety: make sure root included
        if not gated_mask[self.graph.root_node]:
            raise ValueError('Root is masked.')
        
        # safety: ensure at least one fixation available
        if not np.any(gated_mask):
            raise ValueError('All fixation actions are masked.')
        
        mask[:self.num_nodes] = gated_mask
        
        return mask
    

    def set_random_seed(self, seed):
        """
        Set random seed.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)


    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code nodes.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot

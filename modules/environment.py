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
            num_nodes = 11,
            beta_move = 4.0,
            eps_move = 0.02,
            learning_rate = 0.2,
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
        )

        # set random seed
        self.set_random_seed(seed)

        # initialize point set
        self.point_set = self.graph.point_set

        # initialize action space
        self.action_space = Discrete(self.num_nodes + 1)

        # initialize observation space
        observation_shape = (
            self.num_nodes + # q values
            self.num_nodes + # g values
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
        done = False
        reward = -self.cost # initialize reward as cost

        # fixation
        if action < self.num_nodes:
            # debugging
            if action not in self.planner.parents.keys():
                raise ValueError('Fixated node not in the partial tree.')

            # fixate
            self.planner.look(action)

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

        # initialize the tree
        self.graph.reset(shuffle_nodes = self.shuffle_nodes)

        # initialize the planner
        self.planner.init_problem(
            children = self.graph.child_dict,
            rewards = self.graph.point_dict,
            root = self.graph.root_node
        )


    def get_obs(self):
        """
        Get observation.
        """

        # wrap observation
        obs = np.hstack([
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

        # candidates
        for node in self.planner.parents.keys():
            if node != self.planner.root:
                mask[node] = True
        
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

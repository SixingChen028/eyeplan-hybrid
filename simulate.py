import numpy as np
import random
import torch
import warnings
warnings.filterwarnings('ignore')

from modules import *


if __name__ == '__main__':

    # set random seed
    seed = 15
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # parse args
    parser = ArgParser()
    args = parser.args

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.learning_rate}_{args.wm_decay}_{args.jobid}')

    # load net
    net = torch.load(os.path.join(exp_path, f'net.pth'), weights_only = False)

    # set environment
    env = DecisionTreeEnv(
        num_nodes = args.num_nodes,
        beta_move = args.beta_move,
        eps_move = args.eps_move,
        learning_rate = args.learning_rate,
        wm_decay = args.wm_decay,
        t_max = args.t_max,
        cost = args.cost,
        scale_factor = args.scale_factor,
        shuffle_nodes = args.shuffle_nodes,
        mask_fixation = args.mask_fixation,
    )

    # simulate
    num_trials = 100000
    data = simulate(
        net = net,
        env = env,
        num_trials = num_trials,
        greedy = False,
    )
    save_data(data, os.path.join(exp_path, f'data_simulation.p'))




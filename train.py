import os
import gymnasium as gym
import random

from modules import *


if __name__ == '__main__':

    # parse args
    parser = ArgParser()
    args = parser.args

    random.seed(args.seed)

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.learning_rate}_{args.lamda_backup}_{args.wm_decay}_{args.jobid}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # set environment
    seeds = [random.randint(0, 1000) for _ in range(args.batch_size)]
    env = gym.vector.SyncVectorEnv([
        lambda: DecisionTreeEnv(
            num_nodes = args.num_nodes,
            beta_move = args.beta_move,
            eps_move = args.eps_move,
            learning_rate = args.learning_rate,
            lamda_backup = args.lamda_backup,
            wm_decay = args.wm_decay,
            t_max = args.t_max,
            cost = args.cost,
            scale_factor = args.scale_factor,
            shuffle_nodes = args.shuffle_nodes,
            mask_fixation = args.mask_fixation,
            seed = seeds[i],
        )
        for i in range(args.batch_size)
    ])


    # set net
    net = SharedFeedForwardActorCriticPolicy(
        feature_size = env.single_observation_space.shape[0],
        action_size = env.single_action_space.n,
        hidden_size = args.hidden_size,
    )

    # set model
    model = FeedForwardBatchMaskA2C(
        net = net,
        env = env,
        lr = args.lr,
        batch_size = args.batch_size,
        max_grad_norm = args.max_grad_norm,
        gamma = args.gamma,
        lamda = args.lamda,
        beta_v = args.beta_v,
        beta_e = args.beta_e,
        entropy_schedule = np.linspace(
            args.beta_e_init,
            args.beta_e_final,
            int(args.num_episodes / args.batch_size),
        )
    )

    # train network
    data = model.learn(
        num_episodes = args.num_episodes,
        print_frequency = 2000,
    )

    # save net and data
    model.save_net(os.path.join(exp_path, f'net.pth'))
    model.save_data(os.path.join(exp_path, f'data_training.p'))

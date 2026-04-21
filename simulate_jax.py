import os
import pickle

from modules.argument import ArgParser
from modules.jax_a2c import load_jax_params
from modules.jax_environment import JaxDecisionTreeEnv
from modules.jax_simulation import JaxSimulator


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.args

    exp_path = os.path.join(args.path, f'exp_{args.learning_rate}_{args.wm_decay}_{args.jobid}')

    params = load_jax_params(os.path.join(exp_path, 'net_jax.p'))

    env = JaxDecisionTreeEnv(
        num_nodes=args.num_nodes,
        beta_move=args.beta_move,
        eps_move=args.eps_move,
        learning_rate=args.learning_rate,
        wm_decay=args.wm_decay,
        t_max=args.t_max,
        cost=args.cost,
        scale_factor=args.scale_factor,
        shuffle_nodes=args.shuffle_nodes,
    )

    simulator = JaxSimulator(env)
    data = simulator.simulate(
        params=params,
        seed=15,
        num_trials=100000,
        greedy=False,
    )

    with open(os.path.join(exp_path, 'data_simulation_jax.p'), 'wb') as file:
        pickle.dump(data, file)

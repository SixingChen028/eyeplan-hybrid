import os
import pickle

from modules.argument import ArgParser
from modules.jax_run_dirs import resolve_timestamped_run_dir
from modules.jax_a2c import load_jax_params
from modules.jax_environment import JaxDecisionTreeEnv
from modules.jax_simulation import JaxSimulator


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.args
    exp_path = resolve_timestamped_run_dir(path=args.path, jobid=args.jobid)
    print(f"run_dir={exp_path}")

    params = load_jax_params(os.path.join(exp_path, 'net_jax.p'))

    env = JaxDecisionTreeEnv(
        num_nodes=args.num_nodes,
        beta_move=args.beta_move,
        eps_move=args.eps_move,
        learning_rate=args.learning_rate,
        lamda_backup=args.lamda_backup,
        wm_decay=args.wm_decay,
        t_max=args.t_max,
        cost=args.cost,
        scale_factor=args.scale_factor,
        shuffle_nodes=args.shuffle_nodes,
    )

    simulator = JaxSimulator(env)
    data = simulator.simulate(
        params=params,
        seed=args.seed,
        num_trials=100000,
        greedy=False,
    )

    with open(os.path.join(exp_path, 'data_simulation_jax.p'), 'wb') as file:
        pickle.dump(data, file)

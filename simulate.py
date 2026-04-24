import os
import json
import pickle

from modules.argument import ArgParser
from modules.run_dirs import resolve_timestamped_run_dir
from modules.a2c import load_jax_params
from modules.environment import JaxDecisionTreeEnv
from modules.simulation import JaxSimulator, to_transformed_simulation_format


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.args
    exp_path = resolve_timestamped_run_dir(
        path=args.path,
        experiment=args.experiment,
        jobid=args.jobid,
    )
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
        num_trials=args.simulate_trials,
        greedy=False,
    )

    with open(os.path.join(exp_path, 'data_simulation_jax.p'), 'wb') as file:
        pickle.dump(data, file)

    if args.export_transformed_json:
        transformed = to_transformed_simulation_format(
            data,
            num_nodes=args.num_nodes,
            t_max=args.t_max,
            skip_timeout_trials=args.skip_timeout_trials,
        )

        output_path = args.transformed_json_path
        if output_path == '':
            output_path = os.path.join(
                exp_path,
                f'data_{args.learning_rate}_{args.lamda_backup}_{args.wm_decay}_{args.jobid}.json',
            )

        with open(output_path, 'w') as file:
            json.dump(transformed, file)

        print(f"transformed_json={output_path}")
        print(f"transformed_trials={len(transformed['actions'])}")

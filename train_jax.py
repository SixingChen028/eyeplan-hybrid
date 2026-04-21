import os
import pickle
import numpy as np

from modules.argument import ArgParser
from modules.jax_environment import JaxDecisionTreeEnv
from modules.jax_a2c import JaxBatchMaskA2C, save_jax_params


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.args

    exp_path = os.path.join(args.path, f'exp_{args.learning_rate}_{args.wm_decay}_{args.jobid}')
    os.makedirs(exp_path, exist_ok=True)

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

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        lamda=args.lamda,
        beta_v=args.beta_v,
        beta_e=args.beta_e,
    )

    state = trainer.init_state(seed=15)

    num_updates = int(args.num_episodes / args.batch_size)
    entropy_schedule = np.linspace(
        args.beta_e_init,
        args.beta_e_final,
        num_updates,
        dtype=np.float32,
    )

    state, data = trainer.train(
        state=state,
        num_updates=num_updates,
        entropy_schedule=entropy_schedule,
    )

    save_jax_params(state.params, os.path.join(exp_path, 'net_jax.p'))
    with open(os.path.join(exp_path, 'data_training_jax.p'), 'wb') as file:
        pickle.dump(data, file)

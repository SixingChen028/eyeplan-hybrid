import os
import json
import argparse


def parse_bool(value):
    """
    Parse boolean values from CLI strings.
    """

    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n'}:
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use one of true/false/1/0/yes/no."
    )


def parse_recency_decay(value):
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"off", "auto"}:
            return stripped
        value = stripped

    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "recency_decay must be 'off', 'auto', or a number in [0, 1)."
        ) from error

    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError(
            "recency_decay numeric values must satisfy 0 <= recency_decay < 1."
        )
    return parsed


def parse_q_decay(value):
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "auto":
            return stripped
        value = stripped

    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "q_decay must be 'auto' or a number in [0, 1]."
        ) from error

    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            "q_decay numeric values must satisfy 0 <= q_decay <= 1."
        )
    return parsed


class ArgParser:
    """
    An ArgumentParser.
    """

    def __init__(self):
        """
        Initialize the parser.
        """

        # initializa parser
        self.parser = argparse.ArgumentParser()

        # parse arguments
        self.parse_args()
    

    def parse_args(self):
        """
        Parse arguments with default values.

        Note: be careful when editing basic parameters.
        """

        # job parameters
        self.parser.add_argument('--jobid', type = str, default = '', help = 'job id')
        self.parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results'), help = 'path to store results')
        self.parser.add_argument('--experiment', type = str, default = 'default', help = 'experiment name for organizing runs under path/runs/<experiment>')
        self.parser.add_argument('--resume', type = parse_bool, default = False, help = 'resume from most recent matching run (requires jobid)')
        self.parser.add_argument('--seed', type = int, default = 15, help = 'random seed')

        # nework parameters
        self.parser.add_argument('--network_type', type = str, choices = ['mlp', 'node_shared'], default = 'mlp', help = 'network architecture')
        self.parser.add_argument('--hidden_size', type = int, default = 256, help = 'hidden size')

        # environment parameters
        self.parser.add_argument('--num_nodes', type = int, default = 15, help = 'number of nodes')
        self.parser.add_argument('--beta_move', type = float, default = 100.0, help = 'decision temperature')
        self.parser.add_argument('--eps_move', type = float, default = 0.0, help = 'decision lapse rate')
        self.parser.add_argument('--learning_rate', type = float, default = 1.0, help = 'learning_rate')
        self.parser.add_argument('--lamda_backup', type = float, default = 1.0, help = 'backup discount')
        self.parser.add_argument('--backup_steps', type = int, default = 100, help = 'max number of ancestor backups in update_q (0 disables backup)')
        self.parser.add_argument('--wm_decay', type = float, default = 1.0, help = 'working memory decay')
        self.parser.add_argument('--wm_backup', type = parse_bool, default = True, help = 'if backup in update_q should only use active child nodes')
        self.parser.add_argument('--q_drop_rate', type = float, default = 0.0, help = 'probability of resetting q to 0 when node is inactive')
        self.parser.add_argument('--q_drift', type = float, default = 0.0, help = 'inactive q gaussian noise standard deviation per time step')
        self.parser.add_argument('--q_decay', type = parse_q_decay, default = 0.0, help = "inactive q multiplicative decay toward 0, or 'auto'")
        self.parser.add_argument('--t_max', type = int, default = 50, help = 'max time steps per episode')
        self.parser.add_argument('--cost', type = float, default = 0.01, help = 'cost per action')
        self.parser.add_argument('--scale_factor', type = float, default = 1 / 8, help = 'reward scale factor')
        self.parser.add_argument('--shuffle_nodes', type = parse_bool, default = True, help = 'if shuffle nodes')
        self.parser.add_argument('--canonicalize', nargs = '?', const = True, type = parse_bool, default = False, help = 'if canonicalize node ids by discovery order')
        self.parser.add_argument('--recency_decay', type = parse_recency_decay, default = 'auto', help = "off, auto, or numeric decay in [0, 1)")
        self.parser.add_argument('--mask_fixation', type = parse_bool, default = True, help = 'if mask fixations')

        # training parameters
        self.parser.add_argument('--algo', type = str, choices = ['a2c', 'ppo'], default = 'a2c', help = 'training algorithm')
        self.parser.add_argument('--num_updates', type = int, default = 50_000, help = 'number of training updates')
        self.parser.add_argument('--eval_episodes', type = int, default = 102_400, help = 'evaluation episodes')
        self.parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
        self.parser.add_argument('--num_envs', type = int, default = 128, help = 'number of parallel environments')
        self.parser.add_argument('--rollout_length', type = int, default = 50, help = 'environment steps per update (per environment)')
        self.parser.add_argument('--max_grad_norm', type = float, default = 1.0, help = 'gradient clipping')
        self.parser.add_argument('--gamma', type = float, default = 1.0, help = 'temporal discount')
        self.parser.add_argument('--lamda', type = float, default = 0.8, help = 'generalized advantage estimation coefficient')
        self.parser.add_argument('--beta_v', type = float, default = 0.05, help = 'value loss coefficient')
        self.parser.add_argument('--beta_e', type = float, default = 0.05, help = 'entropy regularization coefficient')
        self.parser.add_argument('--beta_e_init', type = float, default = 0.05, help = 'initial entropy regularization coefficient')
        self.parser.add_argument('--beta_e_final', type = float, default = 0.001, help = 'final entropy regularization coefficient')
        self.parser.add_argument('--print_frequency', type = int, default = 100, help = 'print training logs every n updates (0 to disable)')
        self.parser.add_argument('--checkpoint_frequency', type = int, default = 0, help = 'checkpoint cadence: <0 disable, 0 save at each update chunk end, >0 save every n updates')
        self.parser.add_argument('--log_full_metrics', type = parse_bool, default = True, help = 'if collect per-update metrics on host (set false to reduce host sync overhead)')
        self.parser.add_argument('--ppo_epochs', type = int, default = 2, help = 'number of PPO epochs per rollout update')
        self.parser.add_argument('--ppo_clip_eps', type = float, default = 0.2, help = 'PPO clipping epsilon')
        self.parser.add_argument('--ppo_normalize_advantages', type = parse_bool, default = True, help = 'if normalize PPO advantages')

        # parse arguments
        self.args = self.parser.parse_args()
    

    def write_args(self, args_dict):
        """
        Edit arguments.
        """

        for key, value in args_dict.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            else:
                print(f'Warning: {key} is not a valid argument. It will be ignored.')
        
    
    def save_args(self, path):
        """
        Save arguments.
        """

        with open(path, 'w') as f:
            json.dump(vars(self.args), f, indent = 4)

    
    def load_args(self, path):
        """
        Load arguments.
        """

        # error detection
        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: {path}')
        
        # load json
        with open(path, 'r') as f:
            args_dict = json.load(f)

        # write args
        self.write_args(args_dict)

        return self.args



class Args:
    """
    An argument class.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)




if __name__ == '__main__':
    # testing
    parser = ArgParser()
    print(parser.args)

    # parser.save_args(os.path.join(os.getcwd(), 'args.p'))

    # parser.load_args(os.path.join(os.getcwd(), 'args.p'))
    # print(parser.args)

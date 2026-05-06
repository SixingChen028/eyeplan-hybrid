import os
import json
import argparse

DEFAULT_PARAMS = {
    "algo": "a2c",
    "jobid": "",
    "seed": 15,
    "network_type": "mlp",
    "hidden_size": 128,
    "num_nodes": 15,
    "beta_move": 40.0,
    "eps_move": 0.0,
    "learning_rate": 1.0,
    "lamda_backup": 1.0,
    "backup_steps": 100,
    "wm_decay": 1.0,
    "wm_backup": False,
    "q_drop_rate": 0.0,
    "q_drift": 0.0,
    "q_decay": 0.0,
    "t_max": 100,
    "cost": 0.01,
    "scale_factor": 1 / 8,
    "shuffle_nodes": True,
    "recency_decay": "off",
    "mask_fixation": True,
    "num_updates": 31_250,
    "num_envs": 64,
    "rollout_length": 100,
    "eval_episodes": 102_400,
    "lr": 5e-4,
    "max_grad_norm": 1.0,
    "gamma": 1.0,
    "lamda": 0.9,
    "beta_v": 0.05,
    "beta_e": 0.05,
    "beta_e_init": 0.05,
    "beta_e_final": 0.001,
    "print_frequency": 100,
    "checkpoint_frequency": -1,
    "log_full_metrics": True,
}


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
        self.parser.add_argument('--jobid', type = str, default = DEFAULT_PARAMS["jobid"], help = 'job id')
        self.parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results'), help = 'path to store results')
        self.parser.add_argument('--experiment', type = str, default = 'default', help = 'experiment name for organizing runs under path/runs/<experiment>')
        self.parser.add_argument('--resume', type = parse_bool, default = False, help = 'resume from most recent matching run (requires jobid)')
        self.parser.add_argument('--seed', type = int, default = DEFAULT_PARAMS["seed"], help = 'random seed')

        # nework parameters
        self.parser.add_argument('--network_type', type = str, choices = ['mlp', 'node_shared'], default = DEFAULT_PARAMS["network_type"], help = 'network architecture')
        self.parser.add_argument('--hidden_size', type = int, default = DEFAULT_PARAMS["hidden_size"], help = 'hidden size')

        # environment parameters
        self.parser.add_argument('--num_nodes', type = int, default = DEFAULT_PARAMS["num_nodes"], help = 'number of nodes')
        self.parser.add_argument('--beta_move', type = float, default = DEFAULT_PARAMS["beta_move"], help = 'decision temperature')
        self.parser.add_argument('--eps_move', type = float, default = DEFAULT_PARAMS["eps_move"], help = 'decision lapse rate')
        self.parser.add_argument('--learning_rate', type = float, default = DEFAULT_PARAMS["learning_rate"], help = 'learning_rate')
        self.parser.add_argument('--lamda_backup', type = float, default = DEFAULT_PARAMS["lamda_backup"], help = 'backup discount')
        self.parser.add_argument('--backup_steps', type = int, default = DEFAULT_PARAMS["backup_steps"], help = 'max number of ancestor backups in update_q (0 disables backup)')
        self.parser.add_argument('--wm_decay', type = float, default = DEFAULT_PARAMS["wm_decay"], help = 'working memory decay')
        self.parser.add_argument('--wm_backup', type = parse_bool, default = DEFAULT_PARAMS["wm_backup"], help = 'if backup in update_q should only use active child nodes')
        self.parser.add_argument('--q_drop_rate', type = float, default = DEFAULT_PARAMS["q_drop_rate"], help = 'probability of resetting q to 0 when node is inactive')
        self.parser.add_argument('--q_drift', type = float, default = DEFAULT_PARAMS["q_drift"], help = 'inactive q gaussian noise standard deviation per time step')
        self.parser.add_argument('--q_decay', type = parse_q_decay, default = DEFAULT_PARAMS["q_decay"], help = "inactive q multiplicative decay toward 0, or 'auto'")
        self.parser.add_argument('--t_max', type = int, default = DEFAULT_PARAMS["t_max"], help = 'max time steps per episode')
        self.parser.add_argument('--cost', type = float, default = DEFAULT_PARAMS["cost"], help = 'cost per action')
        self.parser.add_argument('--scale_factor', type = float, default = DEFAULT_PARAMS["scale_factor"], help = 'reward scale factor')
        self.parser.add_argument('--shuffle_nodes', type = parse_bool, default = DEFAULT_PARAMS["shuffle_nodes"], help = 'if shuffle nodes')
        self.parser.add_argument('--recency_decay', type = parse_recency_decay, default = DEFAULT_PARAMS["recency_decay"], help = "off, auto, or numeric decay in [0, 1)")
        self.parser.add_argument('--mask_fixation', type = parse_bool, default = DEFAULT_PARAMS["mask_fixation"], help = 'if mask fixations')

        # training parameters
        self.parser.add_argument('--algo', type = str, choices = ['a2c'], default = DEFAULT_PARAMS["algo"], help = 'training algorithm')
        self.parser.add_argument('--num_updates', type = int, default = DEFAULT_PARAMS["num_updates"], help = 'number of training updates')
        self.parser.add_argument('--eval_episodes', type = int, default = DEFAULT_PARAMS["eval_episodes"], help = 'evaluation episodes')
        self.parser.add_argument('--lr', type = float, default = DEFAULT_PARAMS["lr"], help = 'learning rate')
        self.parser.add_argument('--num_envs', type = int, default = DEFAULT_PARAMS["num_envs"], help = 'number of parallel environments')
        self.parser.add_argument('--rollout_length', type = int, default = DEFAULT_PARAMS["rollout_length"], help = 'environment steps per update (per environment)')
        self.parser.add_argument('--max_grad_norm', type = float, default = DEFAULT_PARAMS["max_grad_norm"], help = 'gradient clipping')
        self.parser.add_argument('--gamma', type = float, default = DEFAULT_PARAMS["gamma"], help = 'temporal discount')
        self.parser.add_argument('--lamda', type = float, default = DEFAULT_PARAMS["lamda"], help = 'generalized advantage estimation coefficient')
        self.parser.add_argument('--beta_v', type = float, default = DEFAULT_PARAMS["beta_v"], help = 'value loss coefficient')
        self.parser.add_argument('--beta_e', type = float, default = DEFAULT_PARAMS["beta_e"], help = 'entropy regularization coefficient')
        self.parser.add_argument('--beta_e_init', type = float, default = DEFAULT_PARAMS["beta_e_init"], help = 'initial entropy regularization coefficient')
        self.parser.add_argument('--beta_e_final', type = float, default = DEFAULT_PARAMS["beta_e_final"], help = 'final entropy regularization coefficient')
        self.parser.add_argument('--print_frequency', type = int, default = DEFAULT_PARAMS["print_frequency"], help = 'print training logs every n updates (0 to disable)')
        self.parser.add_argument('--checkpoint_frequency', type = int, default = DEFAULT_PARAMS["checkpoint_frequency"], help = 'checkpoint cadence: <0 disable, 0 save at each update chunk end, >0 save every n updates')
        self.parser.add_argument('--log_full_metrics', type = parse_bool, default = DEFAULT_PARAMS["log_full_metrics"], help = 'if collect per-update metrics on host (set false to reduce host sync overhead)')

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

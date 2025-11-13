import os
import json
import argparse


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
        self.parser.add_argument('--jobid', type = str, default = '0', help = 'job id')
        self.parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results'), help = 'path to store results')

        # nework parameters
        self.parser.add_argument('--hidden_size', type = int, default = 128, help = 'hidden size')

        # environment parameters
        self.parser.add_argument('--num_nodes', type = int, default = 11, help = 'number of nodes')
        self.parser.add_argument('--beta_move', type = float, default = 4.0, help = 'decision temperature')
        self.parser.add_argument('--eps_move', type = float, default = 0.02, help = 'decision lapse rate')
        self.parser.add_argument('--learning_rate', type = float, default = 0.2, help = 'learning_rate')
        self.parser.add_argument('--t_max', type = int, default = 100, help = 'max time steps per episode')
        self.parser.add_argument('--cost', type = float, default = 0.01, help = 'cost per action')
        self.parser.add_argument('--scale_factor', type = float, default = 1 / 8, help = 'reward scale factor')
        self.parser.add_argument('--shuffle_nodes', type = bool, default = True, help = 'if shuffle nodes')
        self.parser.add_argument('--mask_fixation', type = bool, default = True, help = 'if mask fixations')

        # training parameters
        self.parser.add_argument('--num_episodes', type = int, default = 15000000, help = 'training episodes')
        self.parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
        self.parser.add_argument('--batch_size', type = int, default = 40, help = 'batch_size')
        self.parser.add_argument('--max_grad_norm', type = float, default = 1.0, help = 'gradient clipping')
        self.parser.add_argument('--gamma', type = float, default = 1.0, help = 'temporal discount')
        self.parser.add_argument('--lamda', type = float, default = 1.0, help = 'generalized advantage estimation coefficient')
        self.parser.add_argument('--beta_v', type = float, default = 0.05, help = 'value loss coefficient')
        self.parser.add_argument('--beta_e', type = float, default = 0.05, help = 'entropy regularization coefficient')
        self.parser.add_argument('--beta_e_init', type = float, default = 0.05, help = 'initial entropy regularization coefficient')
        self.parser.add_argument('--beta_e_final', type = float, default = 0.02, help = 'final entropy regularization coefficient')

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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    """
    A torch Categorical class with action masking.
    """

    def __init__(self, logits, mask):
        self.mask = mask

        self.mask_value = torch.tensor(
            torch.finfo(logits.dtype).min, dtype = logits.dtype
        )
        logits = torch.where(self.mask, logits, self.mask_value)
        super(CategoricalMasked, self).__init__(logits = logits)


    def entropy(self):
        if self.mask is None:
            return super().entropy()
        
        p_log_p = self.logits * self.probs

        # compute entropy with possible actions only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype = p_log_p.dtype, device = p_log_p.device),
        )

        return -torch.sum(p_log_p, axis = 1)
    

class FlattenExtractor(nn.Module):
    """
    A flatten feature extractor.
    """
    def forward(self, x):
        # keep the first dimension while flatten other dimensions
        return x.view(x.size(0), -1)


class ValueNet(nn.Module):
    """
    Value baseline network.
    """
    
    def __init__(self, input_size):
        super(ValueNet, self).__init__()
        self.fc_value = nn.Linear(input_size, 1)
    
    def forward(self, x):
        value = self.fc_value(x) # (batch_size, 1)

        return value


class ActionNet(nn.Module):
    """
    Action network.
    """

    def __init__(self, input_size, output_size):
        super(ActionNet, self).__init__()
        self.fc_action = nn.Linear(input_size, output_size)
    
    def forward(self, x, mask = None):
        self.logits = self.fc_action(x) # record logits for later analyses

        # no action masking
        if mask == None:
            dist = Categorical(logits = self.logits)
        
        # with action masking
        elif mask != None:
            dist = CategoricalMasked(logits = self.logits, mask = mask)
        
        policy = dist.probs # (batch_size, output_size)
        action = dist.sample() # (batch_size,)
        log_prob = dist.log_prob(action) # (batch_size,)
        entropy = dist.entropy() # (batch_size,)
        
        return action, policy, log_prob, entropy


class SharedFeedForwardActorCriticPolicy(nn.Module):
    """
    GRU recurrent actor-critic policy with shared actor and critic.
    """

    def __init__(
            self,
            feature_size,
            action_size,
            hidden_size = 128,
        ):
        super(SharedFeedForwardActorCriticPolicy, self).__init__()

        # network parameters
        self.feature_size = feature_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # input feature extractor
        self.features_extractor = FlattenExtractor()
        
        # recurrent neural network
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # policy and value net
        self.policy_net = ActionNet(hidden_size, action_size)
        self.value_net = ValueNet(hidden_size)


    def forward(self, obs, mask = None):
        """
        Forward the net.
        """

        # extract input features
        features = self.features_extractor(obs)

        # forward pass
        hidden = F.relu(self.fc2(F.relu(self.fc1(features))))

        # compute action
        action, policy, log_prob, entropy = self.policy_net(hidden, mask)

        # compute value
        value = self.value_net(hidden)

        return action, policy, log_prob, entropy, value


# class SharedFeedForwardActorCriticPolicy(nn.Module):
#     """
#     GRU recurrent actor-critic policy with shared actor and critic.
#     """

#     def __init__(
#             self,
#             feature_size,
#             action_size,
#             hidden_size = 128,
#         ):
#         super(SharedFeedForwardActorCriticPolicy, self).__init__()

#         # network parameters
#         self.feature_size = feature_size
#         self.action_size = action_size
#         self.hidden_size = hidden_size

#         # input feature extractor
#         self.features_extractor = FlattenExtractor()
        
#         # recurrent neural network
#         self.fc1 = nn.Linear(feature_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, hidden_size)

#         # policy and value net
#         self.policy_net = ActionNet(hidden_size, action_size)
#         self.value_net = ValueNet(hidden_size)


#     def forward(self, obs, mask = None):
#         """
#         Forward the net.
#         """

#         # extract input features
#         features = self.features_extractor(obs)

#         # forward pass
#         hidden = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(features))))))

#         # compute action
#         action, policy, log_prob, entropy = self.policy_net(hidden, mask)

#         # compute value
#         value = self.value_net(hidden)

#         return action, policy, log_prob, entropy, value
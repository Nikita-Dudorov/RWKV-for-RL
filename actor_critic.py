import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch import nn

class ContinuousActorCritic(nn.Module):
    """Implements actor-critic agent for row observation and continuous action space"""

    def __init__(self, n_hidden, obs_dim, act_dim):
        super().__init__()

        self._critic = nn.Sequential(
            nn.Linear(obs_dim, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, 1, bias=True)
        )
        
        self._actor_mean = nn.Sequential(
            nn.Linear(obs_dim, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, act_dim, bias=True)
        )

        self._actor_std = nn.Parameter(torch.zeros(1, act_dim))

        # TODO which layer init to use?
        def init_weights(layer, std=1.0, bias=0.0):
            if type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight, std)
                # nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, bias)
        
        self.apply(init_weights)

    def get_value(self, obs):
        return self._critic(obs)

    def get_action(self, obs):
        act_mean = self._actor_mean(obs)
        act_std = self._actor_std.expand_as(act_mean).exp()
        dist = Normal(act_mean, act_std)
        act = dist.sample()
        # return action, action probability, entropy of action distribution
        return act, dist.log_prob(act).sum(1).exp(), dist.entropy().sum(1)  # suppose independetnt components -> summation over action dim 

class DiscreteActorCritic(nn.Module):
    """Implements actor-critic agent for row observation and discrete action space"""

    def __init__(self, n_hidden, obs_dim, act_dim):
        super().__init__()

        self._critic = nn.Sequential(
            nn.Linear(obs_dim, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, 1, bias=True)
        )
        
        self._actor = nn.Sequential(
            nn.Linear(obs_dim, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, act_dim, bias=True)
        )

        # TODO which layer init to use?
        def init_weights(layer, std=1.0, bias=0.0):
            if type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight, std)
                # nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, bias)
        
        self.apply(init_weights)

    def get_value(self, obs):
        return self._critic(obs)

    def get_action(self, obs):
        logits = self._actor(obs)
        dist = Categorical(logits=logits)
        act = dist.sample()
        # return action, action probability, entropy of action distribution
        return act, dist.log_prob(act).exp(), dist.entropy()
import torch
from actor_critic import DiscreteActorCritic
from rwkv import RwkvCell

class RwkvAgent():

    def __init__(
            self,
            d_model,
            d_ac,
            obs_dim,
            act_dim,
        ):
        self.encoder = torch.nn.Linear(obs_dim, d_model)
        self.seq_model = RwkvCell(d_model)
        self.ac = DiscreteActorCritic(
            n_hidden=d_ac, 
            obs_dim=d_model,
            act_dim=act_dim, 
        )
        self.rec_state = None

    def reset_rec_state(self):
        self.rec_state = self.seq_model.get_initial_state()

    def get_hidden(self, obs):
        x = self.encoder(obs)
        x, self.rec_state = self.seq_model(x, self.rec_state)
        return x

    def get_action_and_value(self, obs):
        x = self.get_hidden(obs)
        action = self.ac.get_action(x)
        value = self.ac.get_value(x)
        return action, value

    def get_action(self, obs):
        x = self.get_hidden(obs)
        action = self.ac.get_action(x)
        return action

    def get_value(self, obs):
        x = self.get_hidden(obs)
        value = self.ac.get_value(x)
        return value

from torch import nn
from actor_critic import DiscreteActorCritic
from rwkv import RwkvCell

class RwkvAgent(nn.Module):  
    def __init__(
            self,
            d_model,
            d_ac,
            obs_dim,
            act_dim,
        ):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, d_model)  # supports only row observation
        self.seq_model = RwkvCell(d_model)
        self.ac = DiscreteActorCritic(
            n_hidden=d_ac, 
            obs_dim=d_model,
            act_dim=act_dim, 
        )
        # TODO: init weights

    def reset_rec_state(self):
        new_rec_state = self.seq_model.get_initial_state()
        return new_rec_state

    def get_hidden(self, obs, rec_state):
        x = self.encoder(obs)
        x, new_rec_state = self.seq_model(x, rec_state)
        return x, new_rec_state

    def get_action_and_value(self, obs, rec_state):
        x, new_rec_state = self.get_hidden(obs, rec_state)
        actor_out = self.ac.get_action(x)
        critic_out = self.ac.get_value(x)
        return actor_out, critic_out, new_rec_state

    def get_action(self, obs, rec_state):
        x, new_rec_state = self.get_hidden(obs, rec_state)
        actor_out = self.ac.get_action(x)
        return actor_out, new_rec_state

    def get_value(self, obs, rec_state):
        x, new_rec_state = self.get_hidden(obs, rec_state)
        critic_out = self.ac.get_value(x)
        return critic_out, new_rec_state

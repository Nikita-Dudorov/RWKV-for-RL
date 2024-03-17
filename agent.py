from torch import nn
from actor_critic import DiscreteActorCritic, ContinuousActorCritic
from rwkv import Rwkv

class RwkvAgent(nn.Module):  
    def __init__(
            self,
            n_layers: int,
            d_model: int,
            d_ac: int,
            obs_shape: tuple[int],
            act_dim: int,
            discrete_actions: bool
        ):
        super().__init__()
        assert len(obs_shape) == 1  # currently supports only row observation
        self.encoder = nn.Linear(obs_shape[0], d_model) 
        self.seq_model = Rwkv(n_layers, d_model)
        if discrete_actions:
            self.ac = DiscreteActorCritic(
                n_hidden=d_ac, 
                obs_dim=d_model,
                act_dim=act_dim, 
            )
        else:
            self.ac = ContinuousActorCritic(
                n_hidden=d_ac, 
                obs_dim=d_model,
                act_dim=act_dim, 
            )


    def reset_rec_state(self):
        rec_state = self.seq_model.get_initial_state()
        return rec_state

    def get_hidden(self, obs, rec_state):
        x = self.encoder(obs)
        x, rec_state = self.seq_model(x, rec_state)
        return x, rec_state

    def get_action_and_value(self, obs, rec_state):
        x, rec_state = self.get_hidden(obs, rec_state)
        actor_out = self.ac.get_action(x)
        critic_out = self.ac.get_value(x)
        return actor_out, critic_out, rec_state

    def get_action(self, obs, rec_state, **kwargs):
        x, rec_state = self.get_hidden(obs, rec_state)
        actor_out = self.ac.get_action(x, **kwargs)
        return actor_out, rec_state

    def get_value(self, obs, rec_state):
        x, rec_state = self.get_hidden(obs, rec_state)
        critic_out = self.ac.get_value(x)
        return critic_out, rec_state

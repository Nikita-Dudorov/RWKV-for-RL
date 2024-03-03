import os

class TmazeArgs:
    def __init__(self):
        self.device = 'torch'
        self.seed = 123

        # wandb
        self.project = "RWKV"
        self.tags = ["tmaze_ours"]

        # env
        self.env_id = 'tmaze_ours'
        self.env_config = {
            'episode_length': 1000,
            'corridor_length': 160,
            'goal_reward': 4,
            'goal_penalty': 1,
            'timestep_penalty': 0.1,
            'seed': None,
        }

        # train
        self.n_env_steps = 1e5
        self.n_envs = 8
        self.rollout_len = 256
        self.minibatch = 64
        self.n_epochs = 10
        self.lr = 1e-3
        self.lr_decay = True
        self.max_grad_norm = 0.5

        # log
        self.log_every = 10000
        self.save_every = None
        self.save_path = os.path.join('checkpoints', self.env_id)

        # eval
        self.eval_every = 100000
        self.n_eval_episodes = 100
        self.videos = False

        # agent
        self.d_model = 64
        self.d_ac = 64
        self.ppo_eps = 0.2
        self.ppo_eps_decay = True
        self.gamma = 0.98
        # self.gae_lam = 0.8
        self.c_val_loss = 0.5
        self.c_entr_loss = 0.0
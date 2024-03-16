class TmazeArgs:
    def __init__(self):
        self.device = 'cuda'
        self.seed = 123

        # wandb
        self.project = "RWKV"
        self.tags = ["tmaze_ours"]

        # env
        self.env_id = 'tmaze_ours'
        self.env_config = {
            'episode_length': 250,
            'corridor_length': 40,
            'goal_reward': 4,
            'goal_penalty': 1,
            'timestep_penalty': 0.1,
            'seed': None,
        }

        # train
        self.n_env_steps = 10000000
        self.n_envs = 8
        self.rollout_len = 128
        self.minibatch = 64
        self.n_epochs = 5
        self.lr = 0.0001
        self.lr_decay = 'linear'
        self.max_grad_norm = 0.5

        # log
        self.log_every = 10000
        self.save_every = None
        self.save_path = 'checkpoints/' + self.env_id

        # eval
        self.eval_every = 1000000
        self.n_eval_episodes = 100
        self.videos = False

        # agent
        self.ppo = False
        self.d_model = 64
        self.d_ac = 128
        self.clip_eps = None
        self.clip_eps_decay = False
        self.clip_values = False
        self.norm_adv = False
        self.gamma = 0.99
        self.gae_lam = None
        self.c_val_loss = 0.5
        self.c_entr_loss = 0.01
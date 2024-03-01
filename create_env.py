import gymnasium as gym

def create_env(env_id, **kwargs):
    if env_id == "tmaze_ours":
        from envs.tmaze_ours import TMazeOurs
        env = TMazeOurs(
            episode_length=kwargs["max_episode_steps"],
            corridor_length=kwargs["corridor_len"],
            goal_reward=kwargs["goal_reward"],
            goal_penalty=kwargs["goal_penalty"],
            timestep_penalty=kwargs["timestep_penalty"],
            seed=kwargs["seed"],
        )
        return env
    else:
        raise NotImplementedError(f"Unknow environement '{env_id}'") 

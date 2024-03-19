import gymnasium as gym
from envs.tmaze_ours import TMazeOurs

def create_env(env_id, **kwargs):
    if env_id == "tmaze_ours":
        env = TMazeOurs(
            episode_length=kwargs["episode_length"],
            corridor_length=kwargs["corridor_length"],
            goal_reward=kwargs["goal_reward"],
            goal_penalty=kwargs["goal_penalty"],
            timestep_penalty=kwargs["timestep_penalty"],
            seed=kwargs["seed"],
        )
        return env
    else:
        env = gym.make(env_id, **kwargs)
        return env

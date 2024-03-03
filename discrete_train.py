import os
import random
import torch
import numpy as np
import wandb
import gymnasium as gym
from collections import deque

from create_env import create_env
from config import TmazeArgs
from agent import RwkvAgent


def get_rollout(
        agent, 
        env, 
        init_obs, 
        rollout_len, 
        gamma, 
        device,
    ):
    num_envs, obs_dim = env.observation_space.shape
    observations = torch.zeros((num_envs, rollout_len, obs_dim)).to(device)
    action_probs = torch.zeros((num_envs, rollout_len, 1)).to(device)
    rewards = torch.zeros((num_envs, rollout_len, 1)).to(device)
    rewards_to_go = torch.zeros((num_envs, rollout_len, 1)).to(device)
    values = torch.zeros((num_envs, rollout_len, 1)).to(device)
    advantages = torch.zeros((num_envs, rollout_len, 1)).to(device)
    terminals = np.zeros((num_envs, rollout_len, 1), dtype=np.int32)  # taking action at this step terminates the episode
    starts = np.zeros(num_envs, dtype=np.int32)
    stops = np.zeros(num_envs, dtype=np.int32)
    time_intervals = [[] for n in range(num_envs)]  # first and last steps of episodes within rollout

    # collect policy rollout
    with torch.no_grad():

        init_obs = torch.from_numpy(init_obs).float().to(device)
        actor_out, critic_out = agent.get_action_and_value(init_obs)
        act, act_prob, act_entropy = actor_out
        val = critic_out
        observations[:, 0] = init_obs
        action_probs[:, 0] = act_prob
        values[:, 0] = val

        for t in range(1, rollout_len):
            obs, rwd, done, truncated, info = env.step(act.numpy())  # when using autoreset wrapper, when episode is terminated, obs is a new obs
            terminated = done or truncated
            rewards[:, t-1] = torch.from_numpy(rwd)
            terminals[:, t-1] = terminated
            for n, term in enumerate(terminated):
                if term:
                    stops[n] = t-1
                    time_intervals[n].append((starts[n], stops[n]))
                    starts[n] = t
            obs = torch.from_numpy(obs).float().to(device)
            actor_out, critic_out = agent.get_action_and_value(obs)
            act, act_prob, act_entropy = actor_out
            val = critic_out
            observations[:, t] = obs
            action_probs[:, t] = act_prob 
            values[:, t] = val

        t = rollout_len - 1
        last_obs, rwd, done, truncated, info = env.step(act.numpy())  # when using autoreset wrapper, when episode is terminated, obs is a new obs
        last_val = agent.get_value(torch.from_numpy(last_obs).float().to(device))
        terminated = done or truncated
        rewards[:, t] = torch.from_numpy(rwd)
        terminals[:, t] = terminated
        stops[:] = t
        for n in range(num_envs):
            time_intervals[n].append((starts[n], stops[n]))
        starts[:] = rollout_len

        # calculate rewards-to-go
        for n in range(num_envs):
            for interval in time_intervals[n]:
                start, stop = interval
                for t in reversed(range(start, stop+1)):
                    if terminals[n, t]:
                        rewards_to_go[n, t] = rewards[n, t]
                    else:
                        if t + 1 <= rollout_len - 1:
                            rewards_to_go[n, t] = rewards[n, t] + gamma * rewards_to_go[n, t+1]
                        else:
                            rewards_to_go[n, t] = rewards[n, t] + gamma * last_val[n]
        # rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-8)  # TODO normalize rewards-to-go?
        
        # calculate advantages
        advantages = rewards_to_go - values
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # TODO normalize advantages?
        # TODO gae + lambda

    rollout = {
        'observations': observations,
        'action_probs': action_probs,
        'rewards_to_go': rewards_to_go,
        'values': values,
        'advantages': advantages,
    }
    return rollout, last_obs


def eval(agent, env, n_eval_episodes, device):
    with torch.no_grad():
        scores = torch.zeros(n_eval_episodes)
        for ep in range(n_eval_episodes):
            obs, info = env.reset()
            terminated = False
            ep_ret = 0
            while not terminated:
                act, *_ = agent.get_action(torch.from_numpy(obs).float().to(device))
                obs, rwd, done, truncated, info = env.step(act.item())
                terminated = done or truncated
                ep_ret += rwd
            scores[ep] = ep_ret
    return scores
            

def train():
    args = TmazeArgs()
    ppo_eps = args.ppo_eps
    c_val_loss = args.c_val_loss
    c_entr_loss = args.c_entr_loss
    n_env_steps = args.n_env_steps
    n_envs = args.n_envs
    rollout_len = args.rollout_len
    minibatch = args.minibatch
    batch_size = n_envs * rollout_len
    n_iters = int(n_env_steps / batch_size)
    log_every = int(args.log_every / batch_size)
    eval_every = int(args.eval_every / batch_size)
    save_every = None if args.save_every is None else int(args.save_every / batch_size)
    device = args.device

    # setup wandb
    wandb.init(
        project=args.project,
        tags=args.tags,
        config=args,
        monitor_gym=True,
    )
    # define our custom x axis metric
    wandb.define_metric("env_steps_trained")
    # define which metrics will be plotted against it
    wandb.define_metric("train/loss", step_metric="env_steps_trained")
    wandb.define_metric("train/return_mean", step_metric="env_steps_trained")
    wandb.define_metric("eval/return_mean", step_metric="env_steps_trained")
    wandb.define_metric("eval/return_std", step_metric="env_steps_trained")

    # setup env
    def wrap_env(env):
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.n_eval_episodes)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.NormalizeReward(env)
        return env
    def wrap_eval_env(env):
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.n_eval_episodes)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.NormalizeReward(env)
        if args.videos:
            env = gym.wrappers.RecordVideo(env, video_folder='videos', episode_trigger=lambda k: k % args.n_eval_episodes == 0)
        return env
    env = wrap_env(gym.vector.AsyncVectorEnv([lambda: create_env(args.env_id, **args.env_config) for n in range(n_envs)]))
    eval_env = wrap_eval_env(create_env(args.env_id, **args.env_config))

    # set seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    init_obs, info = env.reset(seed=[random.randint(1, 999) for n in range(n_envs)])

    # define agent
    _, obs_dim = env.observation_space.shape
    act_dim = env.action_space[0].n
    agent = RwkvAgent(
        d_model=args.d_model,
        d_ac=args.d_ac,
        obs_dim=obs_dim,
        act_dim=act_dim,
    ).to(device)
    agent.reset_rec_state() # TODO: agent should have a separate recurrent state for each env and reset it as env resets

    # define optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    # training loop
    run_loss = deque(maxlen=50)
    for iter in range(1, n_iters+1):

        rollout, init_obs = get_rollout(agent, env, init_obs, rollout_len, args.gamma, device)
        observations = rollout['observations'].view(-1, obs_dim)
        action_probs = rollout['action_probs'].view(-1, 1)
        rewards_to_go = rollout['rewards_to_go'].view(-1, 1) 
        # values = rollout['valaues'].view(-1, 1)
        advantages = rollout['advantages'].view(-1, 1)
        assert len(observations) == batch_size

        for epoch in range(args.n_epochs):
            inds = torch.arange(0, batch_size)
            inds = inds[torch.randperm(len(inds))]
            for step in range(batch_size // minibatch):
                b_inds = inds[step*minibatch:(step+1)*minibatch]
                b_obs = observations[b_inds]
                b_act_prob = action_probs[b_inds].view(-1)
                b_rwd_to_go = rewards_to_go[b_inds].view(-1)
                # b_val = values[b_inds].view(-1)
                b_adv = advantages[b_inds].view(-1)
                
                # TODO: normalize batch?
                # b_rwd_to_go = (b_rwd_to_go - b_rwd_to_go.mean()) / (b_rwd_to_go.std() + 1e-8)
                # b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                actor_out, critic_out = agent.get_action_and_value(b_obs)
                b_pred_act, b_pred_act_prob, b_pred_act_entropy = actor_out
                # b_prob_ratio = b_pred_act_prob / b_act_prob
                b_pred_val = critic_out.view(-1)
                # b_pred_val = b_val + (b_pred_val - b_val).clip(-ppo_eps, ppo_eps)  # TODO: clip values?

                actor_loss = -(b_pred_act_prob * b_adv).mean()
                # actor_loss = -torch.min(b_prob_ratio * b_adv, b_prob_ratio.clamp(1-ppo_eps, 1+ppo_eps) * b_adv).mean()  # PPO loss
                value_loss = ((b_pred_val - b_rwd_to_go)**2).mean()
                entropy_loss = -b_pred_act_entropy.mean()

                loss = actor_loss + c_val_loss * value_loss + c_entr_loss * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                run_loss.append(loss.item())

        env_steps_trained = iter * batch_size
        
        if iter % log_every == 0:
            # TODO: running average for all losses
            ep_score_mean = sum(env.return_queue)/(max(len(env.return_queue), 1))
            ep_len_mean = sum(env.length_queue)/(max(len(env.length_queue), 1))
            run_loss_mean = sum(run_loss)/(max(len(run_loss), 1))
            print(f"| iter: {iter} | env steps trained: {env_steps_trained} |")
            print(f"| loss: {loss.item()} | actor loss: {actor_loss.item()} | critic loss: {value_loss.item()} | entropy loss: {entropy_loss.item()} |")
            print(f"| train/return_mean: {ep_score_mean} | train/ep_len_mean: {ep_len_mean} |")
            print(f"| lr: {optimizer.param_groups[0]['lr']} | ppo_eps: {ppo_eps}")
            print()
            wandb.log({'train/loss': run_loss_mean, 'train/return_mean': ep_score_mean, 'env_steps_trained': env_steps_trained})
        if iter % eval_every == 0:
            ep_scores = eval(agent, eval_env, args.n_eval_episodes, device)
            ep_score_mean = ep_scores.mean().item()
            ep_score_std = ep_scores.std().item()
            print(f"| iter: {iter} | env steps trained: {env_steps_trained} |")
            print(f"| eval/return_mean: {ep_score_mean} | eval/return_std: {ep_score_std} |")
            print()
            wandb.log({'eval/return_mean': ep_score_mean, 'eval/return_std': ep_score_std, 'env_steps_trained': env_steps_trained})
        if (save_every is not None) and (iter % save_every == 0):
            torch.save(agent.state_dict(), os.path.join(args.save_path, str(iter)))

        if args.lr_decay:
            optimizer.param_groups[0]['lr'] = args.lr * (1 - iter / n_iters)
        if args.ppo_eps_decay:
            ppo_eps = args.ppo_eps * (1 - iter / n_iters)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
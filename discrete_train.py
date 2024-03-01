import random
import torch
import numpy as np
import wandb
import gymnasium as gym

from create_env import create_env
from config import TmazeArgs
from agent import DiscreteActorCritic
from rwkv import RwkvCell


def get_rollout(agent, env, init_obs, rollout_len, gamma, device, complete_episodes=False):
    obs_dim = env.observation_space.shape[0]
    observations = torch.zeros((rollout_len, obs_dim)).to(device)
    action_probs = torch.zeros((rollout_len, 1)).to(device)
    rewards = torch.zeros((rollout_len, 1)).to(device)
    dones = torch.zeros((rollout_len, 1)).to(device)  # taking action at this step terminates the episode
    rewards_to_go = torch.zeros((rollout_len, 1)).to(device)
    values = torch.zeros((rollout_len, 1)).to(device)
    advantages = torch.zeros((rollout_len, 1)).to(device)
    time_intervals = []  # first and last steps of episodes within rollout
    start = 0

    # collect policy rollout
    with torch.no_grad():

        init_obs = torch.from_numpy(init_obs).float().to(device)
        act, act_prob, act_entropy = agent.get_action(init_obs)
        val = agent.get_value(init_obs)
        observations[0] = init_obs
        action_probs[0] = act_prob
        values[0] = val

        for t in range(1, rollout_len):
            obs, rwd, done, truncated, info = env.step(act.item())
            terminated = done or truncated
            rewards[t-1] = rwd
            dones[t-1] = terminated
            if terminated:
                finish = t-1
                time_intervals.append((start, finish))
                start = t
                obs, info = env.reset()
            obs = torch.from_numpy(obs).float().to(device)
            act, act_prob, act_entropy = agent.get_action(obs)
            val = agent.get_value(obs)
            observations[t] = obs
            action_probs[t] = act_prob 
            values[t] = val

        t = rollout_len - 1
        last_obs, rwd, done, truncated, info = env.step(act.item())
        last_val = agent.get_value(torch.from_numpy(last_obs).float().to(device))
        terminated = done or truncated
        rewards[t] = rwd
        dones[t] = terminated
        finish = t
        time_intervals.append((start, finish))
        start = rollout_len
        if terminated:
            last_obs, info = env.reset()

        # calculate rewards-to-go
        for interval in time_intervals:
            start, finish = interval
            for t in reversed(range(start, finish+1)):
                if dones[t]:
                    rewards_to_go[t] = rewards[t]
                else:
                    if t + 1 <= rollout_len - 1:
                        rewards_to_go[t] = rewards[t] + gamma * rewards_to_go[t+1]
                    else:
                        rewards_to_go[t] = rewards[t] + gamma * last_val
        # rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-8)  # TODO normalize rewards-to-go?
        
        # calculate advantages
        advantages = rewards_to_go - values
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # TODO normalize advantages?
        # TODO gae + lambda

        if complete_episodes: 
            final_step = torch.where(dones == 1)[0].max().item()
            last_obs, info = env.reset()
        else:
            final_step = rollout_len - 1

    return observations[:final_step+1], action_probs[:final_step+1], rewards_to_go[:final_step+1], values[:final_step+1], advantages[:final_step+1], last_obs 


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
            

if __name__ == "__main__":
    args = TmazeArgs()
    ppo_eps = args.ppo_eps
    c_val_loss = args.c_val_loss
    c_entr_loss = args.c_entr_loss
    n_env_steps = args.n_env_steps
    rollout_len = args.rollout_len
    batch_size = args.batch_size
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
    # TODO: vec env
    env = create_env(args.env_id, args.env_config)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.n_eval_episodes)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.NormalizeReward(env)
    eval_env = gym.make(args.env_id, args.env_config)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env, deque_size=args.n_eval_episodes)
    if args.videos:
        eval_env = gym.wrappers.RecordVideo(eval_env, video_folder='videos', episode_trigger=lambda k: k % args.n_eval_episodes == 0)
    # eval_env = gym.wrappers.NormalizeObservation(eval_env)
    # env = gym.wrappers.NormalizeReward(env)

    # set seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.seed(seed)
    # eval_env.seed(seed)

    # define agent and optimizer
    agent = DiscreteActorCritic(
        n_hidden=args.n_hidden, 
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n, 
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    # training loop
    n_iters = int(n_env_steps / rollout_len)
    init_obs, info = env.reset()
    for iter in range(1, n_iters+1):

        *rollout, init_obs = get_rollout(agent, env, init_obs, rollout_len, args.gamma, device)
        observations, action_probs, rewards_to_go, values, advantages = rollout

        for epoch in range(args.n_epochs):
            inds = torch.arange(0, len(observations))
            inds = inds[torch.randperm(len(inds))]
            for step in range(len(observations) // batch_size):
                b_inds = inds[step*batch_size:(step+1)*batch_size]
                b_obs = observations[b_inds]
                b_act_prob = action_probs[b_inds].view(-1)
                b_rwd_to_go = rewards_to_go[b_inds].view(-1)
                # b_val = values[b_inds].view(-1)
                b_adv = advantages[b_inds].view(-1)
                
                # TODO: normalize batch?
                # b_rwd_to_go = (b_rwd_to_go - b_rwd_to_go.mean()) / (b_rwd_to_go.std() + 1e-8)
                # b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                b_pred_val = agent.get_value(b_obs).view(-1)
                # b_pred_val = b_val + (b_pred_val - b_val).clip(-ppo_eps, ppo_eps)  # TODO: clip values?
                b_pred_act, b_pred_act_prob, b_pred_act_entropy = agent.get_action(b_obs)
                # b_prob_ratio = b_pred_act_prob / b_act_prob

                actor_loss = -(b_pred_act_prob * b_adv).mean()
                # actor_loss = -torch.min(b_prob_ratio * b_adv, b_prob_ratio.clamp(1-ppo_eps, 1+ppo_eps) * b_adv).mean()  # PPO loss
                value_loss = ((b_pred_val - b_rwd_to_go)**2).mean()
                entropy_loss = -b_pred_act_entropy.mean()

                loss = actor_loss + c_val_loss * value_loss + c_entr_loss * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        env_steps_trained = iter * rollout_len
        # TODO: running average for loss
        if iter % args.log_every == 0:
            ep_score_mean = sum(env.return_queue)/(max(len(env.return_queue), 1))
            ep_len_mean = sum(env.length_queue)/(max(len(env.length_queue), 1))
            print(f"| iter: {iter} | env steps trained: {env_steps_trained} |")
            print(f"| loss: {loss.item()} | ppo loss: {actor_loss.item()} | value loss: {value_loss.item()} | entropy loss: {entropy_loss.item()} |")
            print(f"| train/return_mean: {ep_score_mean} | train/ep_len_mean: {ep_len_mean} |")
            print(f"| lr: {optimizer.param_groups[0]['lr']} | ppo_eps: {ppo_eps}")
            print()
            wandb.log({'train/loss': loss.item(), 'train/return_mean': ep_score_mean, 'env_steps_trained': env_steps_trained})
        if iter % args.eval_every == 0:
            ep_scores = eval(agent, eval_env, args.n_eval_episodes, device)
            ep_score_mean = ep_scores.mean().item()
            ep_score_std = ep_scores.std().item()
            print(f"| iter: {iter} | env steps trained: {env_steps_trained} |")
            print(f"| eval/return_mean: {ep_score_mean} | eval/return_std: {ep_score_std} |")
            print()
            wandb.log({'eval/return_mean': ep_score_mean, 'eval/return_std': ep_score_std, 'env_steps_trained': env_steps_trained})
        if (args.save_every is not None) and (iter % args.save_every == 0):
            torch.save(agent.state_dict(), args.save_path + '_' + str(iter))

        if args.lr_decay:
            optimizer.param_groups[0]['lr'] = args.lr * (1 - iter / n_iters)
        if args.ppo_eps_decay:
            ppo_eps = args.ppo_eps * (1 - iter / n_iters)

    env.close()
    eval_env.close()

def train():
    state_dim = 10
    d_model = 128
    encoder = torch.nn.Linear(state_dim, d_model)
    model =  RwkvCell(d_model)
    h = model.get_initial_state()
    state = torch.rand(state_dim)
    x = encoder(state)
    x, h = model(x, h)
    print(x.shape, h.shape)

if __name__ == "__main__":
    train()
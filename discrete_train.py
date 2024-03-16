import os
import random
import torch
import numpy as np
import wandb
import yaml
import gymnasium as gym
from collections import deque

from create_env import create_env
from config import TmazeArgs
from agent import RwkvAgent


@torch.no_grad()
def get_rollout(
        agent, 
        agent_state,
        env, 
        init_obs, 
        rollout_len, 
        gamma,
        gae_lam=None,
        device='cpu',
    ):
    num_envs = env.observation_space.shape[0]
    obs_shape = env.observation_space.shape[1:]
    observations = torch.zeros((num_envs, rollout_len, *obs_shape)).to(device)
    action_probs = torch.zeros((num_envs, rollout_len)).to(device)
    rewards = torch.zeros((num_envs, rollout_len)).to(device)
    rewards_to_go = torch.zeros((num_envs, rollout_len)).to(device)
    values = torch.zeros((num_envs, rollout_len)).to(device)
    advantages = torch.zeros((num_envs, rollout_len)).to(device)
    terminals = torch.zeros((num_envs, rollout_len)).to(device)  # taking action at this step terminates the episode
    assert agent_state.shape[0] == num_envs, "agent should have a separate recurrent state for each env"
    rollout_agent_states = torch.zeros((num_envs, rollout_len, *tuple(agent_state.shape[1:]))).to(device)

    # collect policy rollout
    obs = init_obs
    for t in range(0, rollout_len):
        obs = torch.from_numpy(obs).float().to(device)
        actor_out, critic_out, new_agent_state = agent.get_action_and_value(obs, agent_state)
        act, act_prob, act_entropy = actor_out
        val = critic_out.squeeze()
        observations[:, t] = obs
        action_probs[:, t] = act_prob 
        values[:, t] = val
        rollout_agent_states[:, t] = agent_state
        agent_state = new_agent_state

        obs, rwd, done, truncated, info = env.step(act.cpu().numpy())  # when using autoreset wrapper, when episode is terminated, obs is a new obs
        terminated = done | truncated
        for n, term in enumerate(terminated):
            if term: agent_state[n] = agent.reset_rec_state()
        rewards[:, t] = torch.from_numpy(rwd)
        terminals[:, t] = torch.from_numpy(terminated).int()
    last_obs = obs

    # calculate rewards-to-go
    # TODO: rewards_to_go[:, -1] = rewards[:, -1] + gamma * agent.get_value(s_{rollout_len+1})
    rewards_to_go[:, -1] = terminals[:, -1] * rewards[:, -1] + (1 - terminals[:, -1]) * values[:, -1]
    for t in reversed(range(0, rollout_len-1)):
        rewards_to_go[:, t] = rewards[:, t] + gamma * rewards_to_go[:, t+1] * (1 - terminals[:, t])
    
    # calculate advantages
    if gae_lam is None:
        advantages = rewards_to_go - values
    else:
        # TODO: advantages[: -1] = rewards[:, -1] + gamma * agent.get_value(s_{rollout_len+1}) - values[:, -1]
        advantages[:, -1] = rewards[:, -1] - values[:, -1]
        for t in reversed(range(0, rollout_len-1)):
            delta = rewards[:, t] + gamma * values[:, t+1] * (1 - terminals[:, t]) - values[:, t]
            advantages[:, t] = delta + gamma * gae_lam * advantages[:, t+1] * (1 - terminals[:, t])

    rollout = {
        'observations': observations,
        'action_probs': action_probs,
        'rewards_to_go': rewards_to_go,
        'values': values,
        'advantages': advantages,
        'rollout_agent_states': rollout_agent_states,
    }
    return rollout, last_obs, agent_state


@torch.no_grad()
def eval(agent, env, n_eval_episodes, device):
    scores = torch.zeros(n_eval_episodes)
    for ep in range(n_eval_episodes):
        agent_state = agent.reset_rec_state().unsqueeze(0).to(device)  # add batch dim
        obs, info = env.reset()
        terminated = False
        while not terminated:
            actor_out, agent_state = agent.get_action(torch.from_numpy(obs).float().unsqueeze(0).to(device), agent_state)
            act, *_ = actor_out
            obs, rwd, done, truncated, info = env.step(act.item())
            terminated = done or truncated
        if hasattr(env, 'last_success'):
            scores[ep] = int(env.last_success)
        else:
            scores[ep] = env.return_queue[ep]
    return scores
            

def train(args=None):
    clip_eps = args.clip_eps
    c_val_loss = args.c_val_loss
    c_entr_loss = args.c_entr_loss
    n_env_steps = args.n_env_steps
    n_envs = args.n_envs
    rollout_len = args.rollout_len
    minibatch = args.minibatch
    batch_size = n_envs * rollout_len
    n_iters = int(n_env_steps / batch_size)
    log_every = int(args.log_every / batch_size)
    eval_every = None if args.eval_every is None else int(args.eval_every / batch_size)
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
    wandb.define_metric("train/*", step_metric="env_steps_trained")
    wandb.define_metric("eval/*", step_metric="env_steps_trained")

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
    obs_shape = env.observation_space.shape[1:]
    act_dim = env.action_space[0].n
    agent = RwkvAgent(
        d_model=args.d_model,
        d_ac=args.d_ac,
        obs_dim=obs_shape[0],  # supports only row observation
        act_dim=act_dim,
    ).to(device)
    # agent should have a separate recurrent state for each env and reset it as env resets
    agent_states = torch.stack([agent.reset_rec_state() for n in range(n_envs)]).to(device)
    agent_state_shape = tuple(agent_states.shape[1:])

    # define optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    # training loop
    run_window = 50
    run_loss = deque(maxlen=run_window)
    run_actor_loss = deque(maxlen=run_window)
    run_critic_loss = deque(maxlen=run_window)
    run_entropy_loss = deque(maxlen=run_window)
    for iter in range(1, n_iters+1):

        rollout, init_obs, agent_states = get_rollout(agent, agent_states, env, init_obs, rollout_len, args.gamma, args.gae_lam, device)
        observations = rollout['observations'].view(-1, *obs_shape)
        action_probs = rollout['action_probs'].view(-1, 1)
        rewards_to_go = rollout['rewards_to_go'].view(-1, 1) 
        values = rollout['values'].view(-1, 1)
        advantages = rollout['advantages'].view(-1, 1)
        rollout_agent_states = rollout['rollout_agent_states'].view(-1, *agent_state_shape)
        assert len(observations) == batch_size

        # rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-8)  # TODO normalize rewards-to-go?
        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(args.n_epochs):
            inds = torch.arange(0, batch_size)
            inds = inds[torch.randperm(len(inds))]
            for step in range(int(batch_size / minibatch)):
                b_inds = inds[step*minibatch:(step+1)*minibatch]
                b_obs = observations[b_inds]
                b_act_prob = action_probs[b_inds].view(-1)
                b_rwd_to_go = rewards_to_go[b_inds].view(-1)
                b_val = values[b_inds].view(-1)
                b_adv = advantages[b_inds].view(-1)
                b_rollout_agent_st = rollout_agent_states[b_inds]
                
                actor_out, critic_out, pred_agent_st = agent.get_action_and_value(b_obs, b_rollout_agent_st)
                pred_act, pred_act_prob, pred_act_entropy = actor_out
                r = pred_act_prob / b_act_prob
                pred_val = critic_out.view(-1)
                if args.clip_values:
                    pred_val = b_val + (pred_val - b_val).clip(-clip_eps, clip_eps)

                if args.ppo:
                    actor_loss = -torch.min(r * b_adv, r.clip(1-ppo_eps, 1+ppo_eps) * b_adv).mean()
                else:
                    actor_loss = -(torch.log(pred_act_prob) * b_adv).mean()
                critic_loss = c_val_loss * ((pred_val - b_rwd_to_go)**2).mean()
                entropy_loss = -c_entr_loss * pred_act_entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                run_loss.append(loss.item())
                run_actor_loss.append(actor_loss.item())
                run_critic_loss.append(critic_loss.item())
                run_entropy_loss.append(entropy_loss.item())

        env_steps_trained = iter * batch_size
        
        if iter % log_every == 0:
            episode_ret_mean = sum(env.return_queue)/(max(len(env.return_queue), 1))
            episode_len_mean = sum(env.length_queue)/(max(len(env.length_queue), 1))
            loss_mean = sum(run_loss)/len(run_loss)
            actor_loss_mean = sum(run_actor_loss)/len(run_actor_loss)
            critic_loss_mean = sum(run_critic_loss)/len(run_critic_loss)
            entropy_loss_mean = sum(run_entropy_loss)/len(run_entropy_loss)
            print(f"| iter: {iter} | env steps trained: {env_steps_trained} |")
            print(f"| loss: {loss_mean} | actor loss: {actor_loss_mean} | critic loss: {critic_loss_mean} | entropy loss: {entropy_loss_mean} |")
            print(f"| train/episode_ret_mean: {episode_ret_mean} | train/episode_len_mean: {episode_len_mean} |")
            print(f"| lr: {optimizer.param_groups[0]['lr']} | clip_eps: {clip_eps}")
            print("==================")
            wandb.log({
                'train/loss': loss_mean, 
                'train/actor_loss': actor_loss_mean,
                'train/critic_loss': critic_loss_mean,
                'train/entropy_loss': entropy_loss_mean,
                'train/episode_len_mean': episode_len_mean,
                'train/episode_ret_mean': episode_ret_mean,
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/clip_eps': clip_eps, 
                'env_steps_trained': env_steps_trained,
            })
        if (eval_every is not None) and (iter % eval_every == 0):
            episode_scores = eval(agent, eval_env, args.n_eval_episodes, device)
            episode_score_mean = episode_scores.mean().item()
            episode_score_std = episode_scores.std().item()
            print(f"| iter: {iter} | env steps trained: {env_steps_trained} |")
            print(f"| eval/episode_score_mean: {episode_score_mean} | eval/episode_score_std: {episode_score_std} |")
            print("==================")
            wandb.log({
                'eval/episode_len_mean': episode_len_mean,
                'eval/episode_score_mean': episode_score_mean, 
                'eval/episode_score_std': episode_score_std, 
                'env_steps_trained': env_steps_trained
            })
        if (save_every is not None) and (iter % save_every == 0):
            torch.save(agent.state_dict(), os.path.join(args.save_path, str(iter)))

        if args.lr_decay is not None:
            if args.lr_decay == 'linear':
                optimizer.param_groups[0]['lr'] = args.lr * (1 - 0.9*(iter/n_iters))
            elif args.lr_decay == 'exp':
                optimizer.param_groups[0]['lr'] = args.lr * 0.1**(iter/n_iters)
            else:
                raise ValueError(f"unknown decay type: {args.lr_decay}") 
        if args.clip_eps_decay:
            clip_eps = args.clip_eps * (1 - 0.9*(iter/n_iters))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    args = TmazeArgs()
    train(args)

    # sweep mode
    # config_path = './sweep.yaml'
    # project = "RWKV"

    # def sweeper(config=None):
    #     with wandb.init(
    #         config=config,
    #     ):
    #         config = wandb.config
    #         train(config)
            
    # with open(config_path, 'r') as stream:
    #     sweep_config = yaml.safe_load(stream)
    # sweep_id = wandb.sweep(sweep_config, project=project)
    # wandb.agent(sweep_id, sweeper, count=10)
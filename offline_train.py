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


def load_dataset(path):
    dataset = load(path)
    return dataset


def get_batch(
        dataset,
        n_traj_per_batch,
        rollout_len,
        device='cpu',   
    ):
    traj_ids = torch.randint(low=0, high=len(dataset), size=(n_traj_per_batch,))
    batch = dataset[traj_ids, :rollout_len]
    return batch.to(device)


@torch.no_grad()
def eval(agent, env, n_eval_episodes, device):
    scores = torch.zeros(n_eval_episodes)
    lengths = torch.zeros(n_eval_episodes)
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
        lengths[ep] = env.length_queue[ep]
    return {'scores': scores, 'lengths': lengths}
            

def train(args=None):
    n_env_steps = args.n_env_steps
    rollout_len = args.rollout_len
    n_traj_per_batch = args.n_traj_per_batch
    batch_size = n_traj_per_batch * rollout_len 
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
    wandb.define_metric("env_steps_trained")
    wandb.define_metric("train/*", step_metric="env_steps_trained")
    wandb.define_metric("eval/*", step_metric="env_steps_trained")

    # setup env
    def wrap_eval_env(env):
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.n_eval_episodes)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.NormalizeReward(env)
        if args.videos:
            env = gym.wrappers.RecordVideo(env, video_folder='videos', episode_trigger=lambda k: k % args.n_eval_episodes == 0)
        return env
    eval_env = wrap_eval_env(create_env(args.env_id, **args.env_config))

    # set seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # define agent
    obs_shape = eval_env.shape
    act_dim = eval_env.action_space.n
    agent = RwkvAgent(
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_ac=args.d_ac,
        obs_shape=obs_shape,
        act_dim=act_dim,
        discrete_actions=args.discrete_actions
    ).to(device)

    # define optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    # define loss function
    if args.discrete_actions:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    # load dataset
    dataset = load_dataset(args.data_path)

    # training loop
    run_window = 50
    run_loss = deque(maxlen=run_window)
    for iter in range(1, n_iters+1):
        batch = get_batch(dataset, n_traj_per_batch, rollout_len, device)
        assert sum(len(batch[n]['observations']) for n in range(n_traj_per_batch)) == batch_size

        for n in range(n_traj_per_batch):
            traj = batch[n]
            agent_state = agent.reset_rec_state().to(device)
            observations = traj['observations']
            action_probs = traj['actions']
            rewards_to_go = traj['rewards_to_go']
            terminals = traj['terminals'] 
            assert len(observations) == rollout_len
        
            for t in range(rollout_len):
                logits, agent_state = agent.get_action(observations[t], agent_state, return_logits=True)
                loss = loss_fn(logits, action_probs)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                run_loss.append(loss.item())

        env_steps_trained = iter * batch_size
        
        if iter % log_every == 0:
            loss_mean = sum(run_loss)/len(run_loss)
            print(f"| iter: {iter} | env steps trained: {env_steps_trained} |")
            print(f"| loss: {loss_mean} |")
            print(f"| lr: {optimizer.param_groups[0]['lr']} |")
            print("==================")
            wandb.log({
                'train/loss': loss_mean, 
                'train/lr': optimizer.param_groups[0]['lr'],
                'env_steps_trained': env_steps_trained,
            })
        if (eval_every is not None) and (iter % eval_every == 0):
            eval_out = eval(agent, eval_env, args.n_eval_episodes, device)
            episode_scores = eval_out['scores']
            episode_lengths = eval_out['lengths']
            episode_score_mean = episode_scores.mean().item()
            episode_score_std = episode_scores.std().item()
            episode_len_mean = episode_lengths.mean().item()
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
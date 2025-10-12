#!/usr/bin/env python3
"""
Online PPO fine-tuning starting from an offline BC checkpoint.
Minimal, self-contained PPO (actor-critic) using PyTorch.
Designed to be runnable inside your AVs repo and venv.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Try to use optional CLIP encoder/wrapper if available
try:
    from rl_langvision.clip_embedder import CLIPImageEncoder
    from rl_langvision.amb_highway_wrapper_clip import AmbulanceHighwayCLIPWrapper
    HAS_RLLANG = True
except Exception:
    CLIPImageEncoder = None
    AmbulanceHighwayCLIPWrapper = None
    HAS_RLLANG = False


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value


def preprocess_obs(obs, clip_encoder=None, device='cpu'):
    # Convert obs into 1D torch tensor on device
    if isinstance(obs, dict):
        # prefer 'clip' if wrapper provides it
        if 'clip' in obs:
            feat = obs['clip']
            return torch.as_tensor(feat, dtype=torch.float32).to(device)
        # search for numpy array in dict
        for v in obs.values():
            if isinstance(v, np.ndarray):
                arr = v
                break
        else:
            arr = np.asarray(obs)
    else:
        arr = np.asarray(obs)

    if clip_encoder is not None:
        # choose last frame if stacked
        if arr.ndim == 3:
            if arr.shape[0] in (1, 4):
                frame = arr[-1]
            else:
                frame = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            frame = arr
        else:
            frame = arr.squeeze()
        if frame.ndim == 2:
            frame_rgb = np.stack([frame] * 3, axis=-1)
        else:
            frame_rgb = frame
        feat = clip_encoder.encode_np_rgb(frame_rgb)
        return torch.as_tensor(feat, dtype=torch.float32).to(device)

    arr = arr.astype(np.float32)
    return torch.as_tensor(arr.ravel(), dtype=torch.float32).to(device)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    values = np.append(values, last_value)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1.0 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1.0 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    return np.array(returns, dtype=np.float32)


def make_env():
    # Ensure highway_env is imported so it registers its gymnasium entrypoints
    try:
        import highway_env  # registers envs like 'highway-v0'
    except Exception as e:
        print(f"⚠️  Could not import highway_env: {e}")

    if HAS_RLLANG and AmbulanceHighwayCLIPWrapper is not None:
        return AmbulanceHighwayCLIPWrapper({})

    # Try common highway-env ids with clear error messages
    for env_id in ('highway-v0', 'highway-v1', 'highway-v2'):
        try:
            return gym.make(env_id)
        except Exception as e:
            # keep trying next id
            last_err = e

    # If none worked, raise the last error to surface the underlying issue
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/bc_pretrain/best_model.pt')
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--update_epochs', type=int, default=8)
    parser.add_argument('--minibatch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--render', action='store_true', help='Render a few evaluation episodes at the end')
    parser.add_argument('--save-video', action='store_true', help='Save a short evaluation video to --video-dir')
    parser.add_argument('--video-dir', type=str, default='videos', help='Directory to save evaluation videos')
    parser.add_argument('--eval-episodes', type=int, default=2, help='Number of evaluation episodes to render/save')
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision (AMP) when running on CUDA')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Enable cuDNN autotuner for potential speedups on CUDA
    try:
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    use_amp = bool(args.use_amp and device.type == 'cuda')

    env = make_env()
    obs0, _ = env.reset()

    clip_encoder = None
    if CLIPImageEncoder is not None:
        clip_encoder = CLIPImageEncoder(device=str(device))
        obs_feat = preprocess_obs(obs0, clip_encoder=clip_encoder, device=device)
        obs_dim = obs_feat.numel()
    else:
        obs_feat = preprocess_obs(obs0, clip_encoder=None, device=device)
        obs_dim = obs_feat.numel()

    n_actions = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    policy = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    # micro-optimizations
    optimizer_zero_kwargs = {'set_to_none': True} if hasattr(optimizer, 'zero_grad') else {}
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # load checkpoint if exists
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        data = torch.load(str(ckpt), map_location=device)
        sd = None
        if isinstance(data, dict):
            if 'policy_state_dict' in data:
                sd = data['policy_state_dict']
            elif 'state_dict' in data:
                sd = data['state_dict']
            else:
                sd = data
        else:
            sd = data
        try:
            policy.load_state_dict(sd, strict=False)
            print(f'✅ Loaded checkpoint weights from {ckpt}')
        except Exception:
            new = {}
            for k, v in sd.items():
                nk = k.replace('policy.', '').replace('network.', '')
                new[nk] = v
            policy.load_state_dict(new, strict=False)
            print('✅ Loaded adapted checkpoint weights (best-effort)')
    else:
        print('⚠️ Checkpoint not found, training from scratch')

    obs_buffer = []
    actions_buffer = []
    logprobs_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = []

    total_steps = 0
    ep = 0
    start_time = time.time()

    while total_steps < args.timesteps:
        obs_buffer.clear(); actions_buffer.clear(); logprobs_buffer.clear()
        rewards_buffer.clear(); dones_buffer.clear(); values_buffer.clear()
        obs, _ = env.reset()
        for _ in range(args.n_steps):
            feat = preprocess_obs(obs, clip_encoder=clip_encoder, device=device)
            with torch.no_grad():
                logits, value = policy(feat.unsqueeze(0))
                prob = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(prob)
                action = dist.sample().item()
                logp = dist.log_prob(torch.as_tensor(action, device=device))
            obs_buffer.append(feat.cpu().numpy())
            actions_buffer.append(action)
            logprobs_buffer.append(logp.cpu().item())
            values_buffer.append(value.cpu().item())
            out = env.step(action)
            # gymnasium: obs, reward, terminated, truncated, info
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = terminated or truncated
            else:
                obs, reward, done, info = out
            rewards_buffer.append(float(reward))
            dones_buffer.append(bool(done))
            if done:
                obs, _ = env.reset()

        with torch.no_grad():
            last_feat = preprocess_obs(obs, clip_encoder=clip_encoder, device=device)
            _, last_value = policy(last_feat.unsqueeze(0))
            last_value = last_value.cpu().item()

        values = np.array(values_buffer, dtype=np.float32)
        returns = compute_gae(rewards_buffer, values, dones_buffer, last_value, args.gamma, args.gae_lambda)
        advantages = returns - values

        # build tensors on CPU and transfer to device with non_blocking when possible
        obs_tensor = torch.as_tensor(np.stack(obs_buffer), dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions_buffer, dtype=torch.long)
        old_logp = torch.as_tensor(logprobs_buffer, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32)
        try:
            obs_tensor = obs_tensor.to(device, non_blocking=True)
            actions_tensor = actions_tensor.to(device, non_blocking=True)
            old_logp = old_logp.to(device, non_blocking=True)
            returns_tensor = returns_tensor.to(device, non_blocking=True)
            advantages_tensor = advantages_tensor.to(device, non_blocking=True)
        except Exception:
            obs_tensor = obs_tensor.to(device)
            actions_tensor = actions_tensor.to(device)
            old_logp = old_logp.to(device)
            returns_tensor = returns_tensor.to(device)
            advantages_tensor = advantages_tensor.to(device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        batch_size = args.minibatch_size
        idxs = np.arange(len(obs_tensor))
        for epoch in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), batch_size):
                mb_idx = idxs[start:start+batch_size]
                mb_obs = obs_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_oldlogp = old_logp[mb_idx]
                mb_returns = returns_tensor[mb_idx]
                mb_adv = advantages_tensor[mb_idx]

                # forward (mixed precision if enabled)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits, values = policy(mb_obs)
                        probs = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        logp = dist.log_prob(mb_actions)
                        ratio = torch.exp(logp - mb_oldlogp)
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * mb_adv
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.functional.mse_loss(values, mb_returns)
                        entropy = dist.entropy().mean()
                        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                else:
                    logits, values = policy(mb_obs)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    logp = dist.log_prob(mb_actions)
                    ratio = torch.exp(logp - mb_oldlogp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.functional.mse_loss(values, mb_returns)
                    entropy = dist.entropy().mean()
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad(**optimizer_zero_kwargs)
                if use_amp:
                    scaler.scale(loss).backward()
                    # unscale before clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

        total_steps += args.n_steps
        ep += 1
        if ep % 5 == 0:
            out = Path('checkpoints/ppo_online_finetuned.pt')
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_steps': total_steps
            }, str(out))
        elapsed = time.time() - start_time
        print(f'Ep {ep} | Steps {total_steps}/{args.timesteps} | Time {int(elapsed)}s')

    out = Path('checkpoints/ppo_online_finetuned_final.pt')
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_steps': total_steps
    }, str(out))
    print('✅ Finished fine-tuning. Saved to', out)

    # Optional short visual evaluation / video recording
    if args.render or args.save_video:
        print('🔎 Running short evaluation for rendering/video...')
        try:
            # Create a fresh eval env (don't reuse training env to avoid wrappers)
            eval_env = make_env()
            if args.save_video:
                try:
                    video_folder = Path(args.video_dir)
                    video_folder.mkdir(parents=True, exist_ok=True)
                    # gymnasium RecordVideo will handle ffmpeg/video saving if available
                    eval_env = gym.wrappers.RecordVideo(eval_env, str(video_folder))
                except Exception as e:
                    print(f'⚠️  Could not enable RecordVideo wrapper: {e}. Falling back to manual frame capture.')

            for epi in range(args.eval_episodes):
                obs, _ = eval_env.reset()
                done = False
                frames = []
                while not done:
                    feat = preprocess_obs(obs, clip_encoder=clip_encoder, device=device)
                    with torch.no_grad():
                        logits, _ = policy(feat.unsqueeze(0))
                        prob = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(prob)
                        action = int(dist.sample().item())
                    out = eval_env.step(action)
                    if len(out) == 5:
                        obs, reward, terminated, truncated, info = out
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = out

                    if args.render:
                        try:
                            eval_env.render()
                        except Exception:
                            pass

            try:
                eval_env.close()
            except Exception:
                pass
            if args.save_video:
                print(f'✅ Video(s) saved to {args.video_dir} (if RecordVideo succeeded)')
            if args.render:
                print('✅ Render finished')
        except Exception as e:
            print(f'⚠️  Evaluation/rendering failed: {e}')


if __name__ == '__main__':
    main()

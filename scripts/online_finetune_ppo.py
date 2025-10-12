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
try:
    import imageio
except Exception:
    imageio = None

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


def make_env(render_mode=None):
    # Ensure highway_env is imported so it registers its gymnasium entrypoints
    try:
        import highway_env  # registers envs like 'highway-v0'
    except Exception as e:
        print(f"⚠️  Could not import highway_env: {e}")

    if HAS_RLLANG and AmbulanceHighwayCLIPWrapper is not None:
        try:
            return AmbulanceHighwayCLIPWrapper({'render_mode': render_mode} if render_mode else {})
        except Exception:
            return AmbulanceHighwayCLIPWrapper({})

    # Try common highway-env ids with clear error messages
    for env_id in ('highway-v0', 'highway-v1', 'highway-v2'):
        try:
            if render_mode is not None:
                # many gym envs accept a render_mode kwarg
                return gym.make(env_id, render_mode=render_mode)
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
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel envs (vectorized). Uses AsyncVectorEnv when >1')
    parser.add_argument('--debug-vec', action='store_true', help='Print a one-time debug dump of vectorized obs structure')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # GPU diagnostic prints so you can confirm device usage
    print('Device:', device)
    print('torch.cuda.is_available():', torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print('cuda device count:', torch.cuda.device_count())
            print('current device idx:', torch.cuda.current_device())
            print('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception as e:
            print('Could not query CUDA device name:', e)

    # Enable cuDNN autotuner for potential speedups on CUDA
    try:
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    use_amp = bool(args.use_amp and device.type == 'cuda')

    # Create vectorized envs if requested
    if args.num_envs and args.num_envs > 1:
        num_envs = args.num_envs
        try:
            # build async vector env from make_env factory
            def make_fn(i):
                return lambda: make_env()
            env_fns = [make_fn(i) for i in range(num_envs)]
            # When debugging, prefer SyncVectorEnv so exceptions show full tracebacks
            if args.debug_vec:
                env = gym.vector.SyncVectorEnv(env_fns)
            else:
                env = gym.vector.AsyncVectorEnv(env_fns)
        except Exception as e:
            print(f"⚠️  Could not create AsyncVectorEnv: {e}. Falling back to single env.")
            env = make_env()
            num_envs = 1
    else:
        env = make_env()
        num_envs = 1

    obs0, _ = env.reset()

    # If requested, print a one-time compact debug dump of the batched obs structure
    if args.debug_vec and num_envs > 1:
        def _dump_obs_structure(name, o):
            try:
                if isinstance(o, dict):
                    s = {k: (type(v).__name__, getattr(v, 'shape', None)) for k, v in o.items()}
                else:
                    s = (type(o).__name__, getattr(o, 'shape', None))
                print(f"[debug-vec] {name}: {s}")
            except Exception as e:
                print(f"[debug-vec] {name}: <error inspecting: {e}>")

        print('🔍 Debug: vectorized env obs structure (one-time)')
        _dump_obs_structure('obs0', obs0)

    clip_encoder = None
    if CLIPImageEncoder is not None:
        clip_encoder = CLIPImageEncoder(device=str(device))
        # If obs0 is batched (vectorized env), pick first element to infer dim
        if isinstance(obs0, (list, tuple)):
            sample_obs = obs0[0]
        elif isinstance(obs0, dict):
            # batched dict observations have arrays as values with first dim == num_envs
            try:
                sample_obs = {k: (v[0] if hasattr(v, '__getitem__') else v) for k, v in obs0.items()}
            except Exception:
                sample_obs = obs0
        elif (hasattr(obs0, 'shape') and getattr(obs0, 'shape', None) and getattr(obs0, 'shape')[0] == num_envs):
            sample_obs = obs0[0]
        else:
            sample_obs = obs0
        obs_feat = preprocess_obs(sample_obs, clip_encoder=clip_encoder, device=device)
        obs_dim = obs_feat.numel()
    else:
        if isinstance(obs0, (list, tuple)):
            sample_obs = obs0[0]
        elif isinstance(obs0, dict):
            try:
                sample_obs = {k: (v[0] if hasattr(v, '__getitem__') else v) for k, v in obs0.items()}
            except Exception:
                sample_obs = obs0
        elif (hasattr(obs0, 'shape') and getattr(obs0, 'shape', None) and getattr(obs0, 'shape')[0] == num_envs):
            sample_obs = obs0[0]
        else:
            sample_obs = obs0
        obs_feat = preprocess_obs(sample_obs, clip_encoder=None, device=device)
        obs_dim = obs_feat.numel()

    n_actions = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    policy = ActorCritic(obs_dim, n_actions).to(device)
    # Debug: print action space and policy head size
    try:
        head_out = policy.policy[-1].out_features
    except Exception:
        head_out = None
    print(f'Action space n_actions={n_actions} | policy final head out_features={head_out}')
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    # micro-optimizations
    optimizer_zero_kwargs = {'set_to_none': True} if hasattr(optimizer, 'zero_grad') else {}
    # Construct GradScaler using the newer torch.amp API when available to avoid deprecation warnings
    scaler = None
    if use_amp:
        try:
            # preferred API in newer torch: torch.amp.GradScaler(device_type='cuda')
            scaler = torch.amp.GradScaler(device_type=getattr(device, 'type', 'cuda'))
        except Exception:
            # try to call torch.amp.GradScaler without args first, then fall back
            try:
                scaler = torch.amp.GradScaler()
            except Exception:
                try:
                    # older torch versions may only have torch.cuda.amp.GradScaler
                    # prefer passing device_type if supported
                    scaler = torch.cuda.amp.GradScaler()
                except Exception:
                    scaler = None

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

    # helper to extract single-env observation from a batched observation (supports dicts)
    def _extract_obs_from_batch(obs_batch, idx: int):
        if isinstance(obs_batch, dict):
            single = {}
            for k, v in obs_batch.items():
                try:
                    single[k] = v[idx]
                except Exception:
                    single[k] = v
            return single
        # lists/tuples and numpy arrays support indexing
        try:
            return obs_batch[idx]
        except Exception:
            return obs_batch


    while total_steps < args.timesteps:
        obs_buffer.clear(); actions_buffer.clear(); logprobs_buffer.clear()
        rewards_buffer.clear(); dones_buffer.clear(); values_buffer.clear()
        # for vectorized envs obs is batched
        # no need to reset here; env handles ongoing episodes
        epoch_start = time.time()
        for _ in range(args.n_steps):
            # obs may be a batch if num_envs>1
            if num_envs > 1:
                # preprocess each env's obs into feature vector
                feats = []
                for i in range(num_envs):
                    o = _extract_obs_from_batch(obs0, i)
                    f = preprocess_obs(o, clip_encoder=clip_encoder, device=device)
                    feats.append(f)
                feat_batch = torch.stack(feats, dim=0).to(device)
                with torch.no_grad():
                    logits, values = policy(feat_batch)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    actions = dist.sample().cpu().numpy()
                    logps = dist.log_prob(torch.as_tensor(actions, device=device)).cpu().numpy()
                # Sanity-check actions against detected action space size
                try:
                    expected_n = n_actions
                except Exception:
                    expected_n = None
                if expected_n is not None:
                    max_a = int(np.max(actions))
                    min_a = int(np.min(actions))
                    if max_a >= expected_n or min_a < 0:
                        print(f"⚠️  Debug: sampled actions out of range for env.action_space (expected 0..{expected_n-1}): min={min_a} max={max_a}")
                        try:
                            print('    env.action_space:', env.action_space)
                        except Exception:
                            pass
                        # Clip to valid range to avoid KeyError in env.step
                        actions = np.clip(actions, 0, expected_n - 1)
                        print('    ⚠️  Actions clipped to valid range (one-time notice)')
                # Ensure integer dtype for actions before stepping
                try:
                    actions = np.asarray(actions, dtype=np.int64)
                except Exception:
                    actions = np.asarray(actions)
                # Convert to plain Python ints list to avoid numpy-scalar key lookup issues in some envs
                try:
                    actions = [int(x) for x in actions]
                except Exception:
                    pass
                # store per-env
                for i in range(num_envs):
                    obs_buffer.append(feats[i].cpu().numpy())
                    actions_buffer.append(int(actions[i]))
                    logprobs_buffer.append(float(logps[i]))
                    values_buffer.append(float(values[i].cpu().item()))
                out = env.step(actions)
                # env.step returns batched (obs, rewards, terminated, truncated, infos) or (obs, rewards, dones, infos)
                if len(out) == 5:
                    obs0, reward, terminated, truncated, infos = out
                    done = np.logical_or(terminated, truncated)
                else:
                    obs0, reward, done, infos = out
                # append rewards and dones per env
                for r, d in zip(reward, done):
                    rewards_buffer.append(float(r))
                    dones_buffer.append(bool(d))
            else:
                feat = preprocess_obs(obs0, clip_encoder=clip_encoder, device=device)
                with torch.no_grad():
                    logits, value = policy(feat.unsqueeze(0))
                    prob = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(prob)
                    action = dist.sample().item()
                    logp = dist.log_prob(torch.as_tensor(action, device=device))
                # single-env sanity check
                try:
                    expected_n = n_actions
                except Exception:
                    expected_n = None
                if expected_n is not None:
                    if int(action) < 0 or int(action) >= expected_n:
                        print(f"⚠️  Debug: sampled single action out of range for env.action_space (expected 0..{expected_n-1}): action={action}")
                        try:
                            print('    env.action_space:', env.action_space)
                        except Exception:
                            pass
                        # clamp
                        action = int(max(0, min(expected_n - 1, int(action))))
                obs_buffer.append(feat.cpu().numpy())
                actions_buffer.append(action)
                logprobs_buffer.append(logp.cpu().item())
                values_buffer.append(value.cpu().item())
                out = env.step(action)
                # gymnasium: obs, reward, terminated, truncated, info
                if len(out) == 5:
                    obs0, reward, terminated, truncated, info = out
                    done = terminated or truncated
                else:
                    obs0, reward, done, info = out
                rewards_buffer.append(float(reward))
                dones_buffer.append(bool(done))
                if done:
                    obs0, _ = env.reset()

        # Compute last values and per-environment returns when using vectorized envs
        if num_envs > 1:
            # get last values per env from current obs0
            last_feats = []
            for i in range(num_envs):
                o = _extract_obs_from_batch(obs0, i)
                last_feats.append(preprocess_obs(o, clip_encoder=clip_encoder, device=device))
            last_feat_batch = torch.stack(last_feats, dim=0).to(device)
            with torch.no_grad():
                _, last_values_tensor = policy(last_feat_batch)
            last_values = last_values_tensor.cpu().numpy()

            rewards_arr = np.array(rewards_buffer, dtype=np.float32).reshape(args.n_steps, num_envs)
            values_arr = np.array(values_buffer, dtype=np.float32).reshape(args.n_steps, num_envs)
            dones_arr = np.array(dones_buffer, dtype=np.bool_).reshape(args.n_steps, num_envs)

            # compute returns per env and then flatten in the same interleaved order
            returns_per_env = []
            for env_i in range(num_envs):
                r = compute_gae(rewards_arr[:, env_i], values_arr[:, env_i], dones_arr[:, env_i], float(last_values[env_i]), args.gamma, args.gae_lambda)
                returns_per_env.append(r)
            # stack as shape (n_steps, num_envs) then flatten row-major to match storage order
            returns = np.stack(returns_per_env, axis=1).reshape(-1, order='C')
            values = values_arr.reshape(-1, order='C')
            advantages = returns - values
        else:
            with torch.no_grad():
                last_feat = preprocess_obs(obs0, clip_encoder=clip_encoder, device=device)
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
                    # use the newer torch.amp.autocast API when available
                    try:
                        autocast_ctx = torch.amp.autocast(device_type=getattr(device, 'type', 'cuda'))
                    except Exception:
                        autocast_ctx = torch.cuda.amp.autocast()
                    with autocast_ctx:
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

        # total steps increased by n_steps * num_envs
        total_steps += args.n_steps * num_envs
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
        # compute epoch elapsed and steps/sec for this epoch
        epoch_elapsed = time.time() - epoch_start
        # when using vectorized envs, steps per second is per-env steps/sec times num_envs
        steps_per_sec = (args.n_steps * num_envs) / epoch_elapsed if epoch_elapsed > 0 else float('inf')
        # Per-epoch GPU memory usage (if CUDA enabled)
        if torch.cuda.is_available():
            try:
                alloc_mb = torch.cuda.memory_allocated() // 1024 ** 2
                resv_mb = torch.cuda.memory_reserved() // 1024 ** 2
                print(f'Ep {ep} | Steps {total_steps}/{args.timesteps} | Time {int(elapsed)}s | Epoch time {epoch_elapsed:.2f}s | {steps_per_sec:.1f} steps/s | GPU mem alloc {alloc_mb}MB reserved {resv_mb}MB')
            except Exception:
                print(f'Ep {ep} | Steps {total_steps}/{args.timesteps} | Time {int(elapsed)}s | Epoch time {epoch_elapsed:.2f}s | {steps_per_sec:.1f} steps/s | GPU mem info unavailable')
        else:
            print(f'Ep {ep} | Steps {total_steps}/{args.timesteps} | Time {int(elapsed)}s | Epoch time {epoch_elapsed:.2f}s | {steps_per_sec:.1f} steps/s')

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
            # If saving video, request an env that returns RGB frames
            if args.save_video:
                eval_env = make_env(render_mode='rgb_array')
            else:
                eval_env = make_env()
            if args.save_video:
                video_folder = Path(args.video_dir)
                video_folder.mkdir(parents=True, exist_ok=True)
                try:
                    # gymnasium RecordVideo will record if env supports rgb_array render mode
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
                    # Manual frame capture fallback: if RecordVideo couldn't be used, try to collect frames
                    if args.save_video and imageio is not None:
                        try:
                            frm = None
                            try:
                                # some envs return frames from render()
                                frm = eval_env.render()
                            except Exception:
                                # other envs may provide last rendered frame in info or require different call
                                frm = None
                            if frm is not None:
                                frames.append(frm)
                        except Exception:
                            pass

            try:
                eval_env.close()
            except Exception:
                pass
            # If we collected frames manually and imageio is available, write mp4 files
                if args.save_video:
                    if imageio is None:
                        print('⚠️  imageio not installed; cannot write manual video. Install imageio[ffmpeg] to enable.')
                    else:
                        # Write one MP4 per episode if frames were collected
                        if len(frames) > 0:
                            fname = Path(args.video_dir) / f'ppo_eval_ep{epi + 1}.mp4'
                            try:
                                # ensure frames are uint8 images
                                arrs = [(f.astype('uint8') if hasattr(f, 'astype') else f) for f in frames]
                                imageio.mimwrite(str(fname), arrs, fps=30)
                                print(f'✅ Wrote manual video to {fname}')
                            except Exception as e:
                                print(f'⚠️  Failed to write video {fname}: {e}')
                    print(f'✅ Video(s) saved to {args.video_dir} (if RecordVideo succeeded)')
            if args.render:
                print('✅ Render finished')
        except Exception as e:
            print(f'⚠️  Evaluation/rendering failed: {e}')


if __name__ == '__main__':
    main()

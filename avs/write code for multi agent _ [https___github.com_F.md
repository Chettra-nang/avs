<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# write code for multi agent : [https://github.com/Farama-Foundation/HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)

Here’s a ready-to-run template showing how to configure HighwayEnv for true multi-agent control (multiple controllable vehicles), with tuple observations/actions, plus minimal training/evaluation loops for centralized and per-agent policies.

Quick start

- pip install highway-env gymnasium numpy[^1][^2]

Multi-agent environment setup

```python
import gymnasium as gym
import highway_env
import numpy as np

def make_ma_env(
    env_id="highway-v0",
    n_agents=2,
    vehicles_count=20,
    render_mode=None,
    seed=0,
):
    # Create base env
    env = gym.make(
        env_id,
        render_mode=render_mode or "rgb_array",
        config={
            # multiple controlled vehicles
            "controlled_vehicles": n_agents,
            "vehicles_count": vehicles_count,
            # multi-agent observation: tuple of per-agent obs
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {  # specify per-agent observation type
                    "type": "Kinematics",  # e.g., Kinematics matrix
                    # Optional Kinematics config:
                    # "features": ["presence","x","y","vx","vy","heading"],
                    # "absolute": False,
                    # "normalize": True,
                    # "vehicles_count": 5,
                },
            },
            # multi-agent action: tuple of per-agent actions
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",  # left, keep, right, faster, slower
                    # Optional DiscreteMetaAction config:
                    # "speed_delta": 5.0,
                    # "lane_change_duration": 1.0,
                },
            },
            # Typical highway reward shaping
            "reward_speed_range": [20, 30],
            "duration": 40,
            "lanes_count": 4,
            "policy_frequency": 15,
            "vehicles_density": 1.0,
        },
    )
    obs, info = env.reset(seed=seed)
    return env

if __name__ == "__main__":
    env = make_ma_env(n_agents=2, render_mode="rgb_array")
    obs, info = env.reset()
    # obs is a tuple of per-agent observations (e.g., each a Kinematics matrix)
    print("Num agents:", len(obs))
```

Step with tuple actions

```python
# DiscreteMetaAction: typical mapping is:
# 0: LANE_LEFT, 1: IDLE/KEEP, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
# See highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL for exact list.
# Example: make agent 0 go left, agent 1 go right
actions = (0, 2)
obs, reward, terminated, truncated, info = env.step(actions)
```

Centralized policy loop (single net over stacked observations)

```python
def central_policy(obs_tuple):
    # Simple heuristic: keep lane for all agents
    n = len(obs_tuple)
    return tuple([^1] * n)  # IDLE/KEEP for all

episodes = 3
for ep in range(episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    ep_reward = 0.0
    while not (terminated or truncated):
        act = central_policy(obs)
        obs, reward, terminated, truncated, info = env.step(act)
        # reward is scalar (environment-level) in HighwayEnv; aggregate as desired
        ep_reward += reward
    print(f"Episode {ep} reward: {ep_reward}")
```

Decentralized per-agent policy loop

```python
def per_agent_policy(obs):
    # obs: per-agent Kinematics matrix (V x F)
    # trivial heuristic: accelerate if possible
    return 3  # FASTER

episodes = 3
for ep in range(episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    ep_reward = 0.0
    while not (terminated or truncated):
        actions = tuple(per_agent_policy(o_i) for o_i in obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        ep_reward += reward
    print(f"[Per-agent] Episode {ep} reward: {ep_reward}")
```

Training adapter (PPO-style centralized policy skeleton)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CentralPPO(nn.Module):
    def __init__(self, obs_dim, n_agents, n_actions):
        super().__init__()
        # Example: flatten and concat all agents’ Kinematics matrices
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agents, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(128, n_actions * n_agents)  # per-agent logits
        self.v_head = nn.Linear(128, 1)

    def forward(self, obs_tuple):
        # obs_tuple: tuple of arrays; each is (V,F). Flatten per agent then concat
        flats = [torch.tensor(o, dtype=torch.float32).flatten() for o in obs_tuple]
        x = torch.cat(flats, dim=0).unsqueeze(0)  # batch=1
        h = self.net(x)
        logits = self.pi_head(h).view(-1, len(obs_tuple), self.n_actions)  # [B, A, Actions]
        value = self.v_head(h).squeeze(-1)
        return logits, value

def sample_action_from_logits(logits):
    # logits: [1, n_agents, n_actions]
    acts = []
    for i in range(logits.shape[^1]):
        dist = torch.distributions.Categorical(logits=logits[0, i])
        acts.append(int(dist.sample().item()))
    return tuple(acts)

# Discover shapes from a reset
env = make_ma_env(n_agents=2)
obs, info = env.reset()
n_agents = len(obs)

# Infer per-agent obs flat size from one sample
o0 = obs[^0]
obs_dim = int(np.prod(o0.shape))  # V*F for Kinematics

n_actions = 5  # DiscreteMetaAction default
agent = CentralPPO(obs_dim=obs_dim, n_agents=n_agents, n_actions=n_actions)
optimizer = optim.Adam(agent.parameters(), lr=3e-4)

# Minimal rollouts
for ep in range(2):
    obs, info = env.reset()
    terminated = truncated = False
    traj = []
    while not (terminated or truncated):
        logits, value = agent(obs)
        action = sample_action_from_logits(logits)
        next_obs, reward, terminated, truncated, info = env.step(action)
        traj.append((obs, action, reward, value.detach()))
        obs = next_obs
    print("Collected steps:", len(traj))
```

Key API notes

- Multi-agent setup requires both the MultiAgentAction and MultiAgentObservation wrappers in the config; otherwise only the first vehicle will be controlled/observed.[^3]
- Kinematics observation returns a $V \times F$ array per agent; configure features/normalization/vehicles_count as needed.[^4]
- DiscreteMetaAction provides 5 high-level controls: left, keep, right, faster, slower; send a tuple with one discrete action per controlled vehicle.[^3]

Where to go next

- User guide “The Multi-Agent setting” shows increasing controlled_vehicles, switching action/observation to multi-agent, and stepping with tuples.[^1][^3]
- Observation configuration options for Kinematics and others are documented here.[^4]
- HighwayEnv package page for installation and supported envs.[^2]
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^5][^6][^7][^8][^9]</span>

<div style="text-align: center">⁂</div>

[^1]: https://highway-env.farama.org/user_guide/

[^2]: https://pypi.org/project/highway-env/

[^3]: https://highway-env.farama.org/multi_agent/

[^4]: https://highway-env.farama.org/observations/index.html

[^5]: https://github.com/Farama-Foundation/HighwayEnv

[^6]: https://claudezss.com/project/multi-agent-cooperative-driving-in-highwayenv

[^7]: https://highway-env.farama.org/make_your_own/

[^8]: https://real.mtak.hu/175046/1/IEEE_CogMob_2023_paper_15.pdf

[^9]: https://arxiv.org/html/2506.11445v1

[^10]: https://arxiv.org/html/2403.17805v2

[^11]: https://arxiv.org/pdf/2506.11445.pdf

[^12]: https://arxiv.org/html/2111.06318v2

[^13]: https://www.sciencedirect.com/science/article/pii/S1110016824017095

[^14]: https://github.com/eleurent/highway-env/issues/404

[^15]: https://liralab.usc.edu/pdfs/publications/haynam2025multi.pdf

[^16]: https://github.com/eleurent/highway-env/issues/35

[^17]: https://www.gymlibrary.dev/environments/third_party_environments/

[^18]: https://stable-baselines3.readthedocs.io/en/v2.0.0/misc/projects.html

[^19]: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

[^20]: https://www.diva-portal.org/smash/get/diva2:1573441/FULLTEXT01.pdf


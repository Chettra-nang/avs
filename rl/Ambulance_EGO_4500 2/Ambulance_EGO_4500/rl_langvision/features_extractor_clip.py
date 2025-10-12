from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CLIPLangExtractor(BaseFeaturesExtractor):
    """
    Observation space: Dict({"clip": Box(..., shape=(Dv,)),
                             "text": Box(..., shape=(Dt,))})
    Output: features_dim
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512,
                 proj_hidden: int | None = None, drop_p: float = 0.1):
        clip_shape = observation_space.spaces["clip"].shape
        text_shape = observation_space.spaces["text"].shape
        clip_dim = int(np.prod(clip_shape))
        text_dim = int(np.prod(text_shape))

        super().__init__(observation_space, features_dim)

        h = proj_hidden or max(256, features_dim)

        # Per-modality projection to a common size (features_dim//2 each)
        half = max(128, features_dim // 2)

        self.clip_proj = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, half),
            nn.SiLU(),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, half),
            nn.SiLU(),
        )

        # Fusion + residual head
        self.fuse = nn.Sequential(
            nn.LayerNorm(half * 2),
            nn.Linear(half * 2, h),
            nn.SiLU(),
            nn.Dropout(drop_p),
            nn.Linear(h, features_dim),
        )
        self.res = nn.Sequential(
            nn.LayerNorm(half * 2),
            nn.Linear(half * 2, features_dim),
        )

        # init a bit conservatively
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Expect float32; if not, cast (cheap)
        clip = obs["clip"].float()
        text = obs["text"].float()

        # If you already L2-normalize upstream, you can skip this:
        # clip = torch.nn.functional.normalize(clip, dim=-1)
        # text = torch.nn.functional.normalize(text, dim=-1)

        hc = self.clip_proj(clip)
        ht = self.text_proj(text)
        z = torch.cat([hc, ht], dim=-1)

        y = self.fuse(z)
        y = y + self.res(z)  # residual
        return y



# from __future__ import annotations
# import numpy as np
# import torch
# import torch.nn as nn
# from gymnasium import spaces
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# class CLIPLangExtractor(BaseFeaturesExtractor):
#     """
#     SB3 FeaturesExtractor for Dict obs: {"clip": (B, Dv), "text": (B, Dt)}.
#     Concatenates the two and projects to features_dim via a small MLP.
#     """
#     def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
#         clip_dim = int(np.prod(observation_space.spaces["clip"].shape))
#         text_dim = int(np.prod(observation_space.spaces["text"].shape))
#         in_dim = clip_dim + text_dim

#         super().__init__(observation_space, features_dim)

#         hidden = max(256, features_dim)
#         self.net = nn.Sequential(
#             nn.LayerNorm(in_dim),
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, features_dim),
#         )

#     def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
#         x = torch.cat([obs["clip"], obs["text"]], dim=-1)
#         return self.net(x)


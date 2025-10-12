# rl_langvision/reward_wrappers.py
from __future__ import annotations
import numpy as np
import gymnasium as gym

def _unit(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

class SafetySpeedRewardWrapper(gym.Wrapper):
    """
    Shaped reward for the ambulance (ego).

      total_r = + speed_w * norm_speed
                + clear_bonus (if no blockers ahead)
                - block_penalty_w * (#blockers ahead)
                - ttc_w * penalty when TTC < threshold
                - collision_penalty (only on crash)

    Notes
    -----
    - Always reads ego/road from the *base* HighwayEnv (self._base()) so it
      still works when other wrappers are stacked above.
    """

    def __init__(self, env: gym.Env, cfg: dict | None = None):
        super().__init__(env)
        self.cfg = {
            "speed_min": 20.0,    # m/s
            "speed_max": 30.0,    # m/s
            "ahead_dist": 35.0,   # m
            "same_lane_only": True,
            "ttc_threshold": 2.0, # s
            "ttc_w": 0.30,
            "speed_w": 1.00,
            "clear_bonus": 0.30,
            "block_penalty_w": 0.15,
            "collision_penalty": 1.00,
            "clip_abs": 1.0,
        }
        if cfg:
            self.cfg.update(cfg)

    # ---------- helpers over base env ----------
    def _base(self):
        """Return the unwrapped base env (HighwayEnv)."""
        env = self.env
        # unwrap through any number of wrappers
        while hasattr(env, "env"):
            env = env.env
        return env

    def _ego(self):
        base = self._base()
        return getattr(base, "vehicle", None) or getattr(base, "ego_vehicle", None)

    def _road(self):
        return getattr(self._base(), "road", None)

    def _same_lane(self, a, b) -> bool:
        try:
            return a.lane_index[2] == b.lane_index[2]
        except Exception:
            return True

    def _blockers_ahead(self, ego) -> int:
        road = self._road()
        if road is None:
            return 0
        ahead = 0
        for v in getattr(road, "vehicles", ()):
            if v is ego:
                continue
            dx = v.position[0] - ego.position[0]
            if dx <= 0.0 or dx > self.cfg["ahead_dist"]:
                continue
            if self.cfg["same_lane_only"] and not self._same_lane(v, ego):
                continue
            ahead += 1
        return ahead

    def _min_ttc(self, ego) -> float:
        road = self._road()
        if road is None:
            return 1e8
        min_ttc = 1e9
        for v in getattr(road, "vehicles", ()):
            if v is ego:
                continue
            dx = v.position[0] - ego.position[0]
            if dx <= 0.0 or dx > self.cfg["ahead_dist"]:
                continue
            if self.cfg["same_lane_only"] and not self._same_lane(v, ego):
                continue
            rel_v = (ego.speed - v.speed) + 1e-6
            if rel_v <= 0:
                continue
            ttc = dx / rel_v
            if ttc < min_ttc:
                min_ttc = ttc
        return min_ttc if min_ttc < 1e8 else 1e8

    # ---------- main ----------
    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}

        ego = self._ego()
        if ego is None:
            # keep keys for TB even if something goes wrong
            info.update({
                "r_speed": 0.0, "r_clear": 0.0, "r_block": 0.0,
                "r_ttc_pen": 0.0, "r_collision": 0.0,
                "blockers_ahead": 0, "ttc_min": float(1e8),
                "speed_kph": 0.0, "crashed": bool(info.get("crashed", False)),
            })
            return obs, base_r, terminated, truncated, info

        # speed
        speed_mps = float(getattr(ego, "speed", 0.0))
        speed_kph = speed_mps * 3.6
        vnorm = _unit(speed_mps, self.cfg["speed_min"], self.cfg["speed_max"])
        r_speed = self.cfg["speed_w"] * vnorm

        # corridor & blockers
        blockers = self._blockers_ahead(ego)
        r_clear = self.cfg["clear_bonus"] if blockers == 0 else 0.0
        r_block = - self.cfg["block_penalty_w"] * float(blockers)

        # TTC penalty
        ttc = self._min_ttc(ego)
        r_ttc_pen = 0.0
        if ttc < self.cfg["ttc_threshold"]:
            r_ttc_pen = self.cfg["ttc_w"] * (1.0 - (ttc / max(self.cfg["ttc_threshold"], 1e-6)))

        # crash
        crashed = bool(info.get("crashed", getattr(ego, "crashed", False)))
        r_collision = - self.cfg["collision_penalty"] if crashed else 0.0

        # total shaped reward (replace env reward)
        shaped = r_speed + r_clear + r_block - r_ttc_pen + r_collision
        total_r = shaped
        ca = self.cfg.get("clip_abs", None)
        if ca is not None:
            total_r = float(np.clip(total_r, -float(ca), float(ca)))

        # log
        info.update({
            "r_speed": float(r_speed),
            "r_clear": float(r_clear),
            "r_block": float(r_block),
            "r_ttc_pen": float(r_ttc_pen),
            "r_collision": float(r_collision),
            "blockers_ahead": int(blockers),
            "ttc_min": float(ttc),
            "speed_kph": float(speed_kph),
            "crashed": crashed,
        })

        return obs, total_r, terminated, truncated, info

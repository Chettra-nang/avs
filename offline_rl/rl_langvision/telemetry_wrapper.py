# rl_langvision/telemetry_wrapper.py
from __future__ import annotations
import gymnasium as gym
import numpy as np

class InjectTelemetryInfo(gym.Wrapper):
    """
    Adds 'speed_kph' and 'ttc_min' into info each step.
    - speed_kph: ego speed (m/s) * 3.6
    - ttc_min: rough TTC to front vehicle (seconds); large cap if none/safe.
    Works even if inner wrappers change observations; reaches HighwayEnv via unwrapped.
    """
    def __init__(self, env: gym.Env, ttc_cap: float = 1e8):
        super().__init__(env)
        self._ttc_cap = float(ttc_cap)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})

        base = self.env.unwrapped  # underlying highway_env environment
        speed_kph = 0.0
        ttc_min = self._ttc_cap

        try:
            ego = getattr(base, "vehicle", None)
            if ego is not None:
                v = float(getattr(ego, "speed", 0.0))
                speed_kph = float(v * 3.6)

                road = getattr(base, "road", None)
                if road is not None and hasattr(road, "neighbour"):
                    try:
                        front = road.neighbour(ego, 1)  # +1 forward neighbour
                    except TypeError:
                        front = None
                    if front is not None and hasattr(front, "speed"):
                        gap = None
                        # Prefer lane-frame longitudinal gap
                        try:
                            lane = road.network.get_lane(ego.lane_index)
                            s_ego = lane.local_coordinates(ego.position)[0]
                            s_front = lane.local_coordinates(front.position)[0]
                            gap = float(s_front - s_ego - ego.LENGTH)
                        except Exception:
                            pass
                        # Fallback: Euclidean distance minus length
                        if gap is None or not np.isfinite(gap):
                            try:
                                diff = np.array(front.position) - np.array(ego.position)
                                gap = float(np.linalg.norm(diff) - ego.LENGTH)
                            except Exception:
                                gap = None

                        if gap is not None and gap > 0:
                            rel_v = float(ego.speed - front.speed)  # >0 = closing in
                            if rel_v > 0.1:
                                ttc = gap / rel_v
                                if np.isfinite(ttc) and ttc > 0.0:
                                    ttc_min = min(ttc_min, ttc)
        except Exception:
            pass

        info["speed_kph"] = float(speed_kph)
        info["ttc_min"] = float(ttc_min)
        return obs, reward, terminated, truncated, info

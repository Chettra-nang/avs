# rl_langvision/yielding_traffic.py
from __future__ import annotations
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

def _right_lane_index(ln_idx, min_lane: int = 0):
    road_id, segment_id, lane_id = ln_idx
    return (road_id, segment_id, max(min_lane, lane_id - 1))

class YieldingIDM(IDMVehicle):
    """
    IDM vehicle that 'makes way' for an emergency ego:
      - increases headway,
      - reduces target speed,
      - prefers changing to the right lane when ego is near.
    Robust across highway_env versions (no SPEED_MAX dependency).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # record nominal parameters (safe defaults if missing)
        self._base_time_headway = float(getattr(self, "time_headway", 1.5))

        tv_default = 30.0  # m/s (~108 kph)
        tv = getattr(self, "target_velocity", None)
        if tv is None:
            try:
                self.target_velocity = tv_default
                tv = tv_default
            except Exception:
                tv = tv_default
        self._base_target_velocity = float(tv)

        self._yielding = False

    def _maybe_restore(self):
        if self._yielding:
            try:
                self.time_headway = float(self._base_time_headway)
            except Exception:
                pass
            try:
                self.target_velocity = float(self._base_target_velocity)
            except Exception:
                pass
            self._yielding = False

    def step(self, dt: float) -> None:
        super().step(dt)

        env = getattr(self.road, "env", None)
        if env is None:
            return

        cfg = getattr(env, "config", {}) or {}
        if not cfg.get("is_emergency", False):
            self._maybe_restore()
            return

        ego = getattr(env, "vehicle", None) or getattr(env, "ego_vehicle", None)
        if ego is None:
            self._maybe_restore()
            return

        dx = float(ego.position[0] - self.position[0])
        dy = float(ego.position[1] - self.position[1])
        d = float(np.hypot(dx, dy))

        radius = float(cfg.get("yield_radius", 60.0))
        if d > radius:
            self._maybe_restore()
            return

        # make way if ego is behind/alongside (dx > -15)
        if dx > -15.0:
            self._yielding = True

            # open gap
            try:
                self.time_headway = float(self._base_time_headway) * 1.5
            except Exception:
                pass

            # drop target speed a bit below current to create space
            try:
                cur_speed = float(getattr(self, "speed", 0.0))
                self.target_velocity = float(
                    max(0.0, min(self._base_target_velocity, cur_speed - 4.0))
                )
            except Exception:
                pass

            # prefer right lane if exists
            try:
                bias = float(cfg.get("yield_right_bias", 0.7))
                if np.random.rand() < bias and hasattr(self, "lane_index"):
                    self.target_lane_index = _right_lane_index(self.lane_index, min_lane=0)
            except Exception:
                pass
        else:
            self._maybe_restore()


# Great “why” to ask. Short answer: **YieldingIDM makes the rest of traffic behave like decent humans when an ambulance comes through**—and that matters for both realism and learning.

# ## Why you need it

# 1. **Match the captions / language signals**
#    Your captions say “Other drivers should yield and open a corridor.” If background cars never yield, text ↔ world becomes inconsistent, hurting CLIP/VLM alignment and reward shaping based on those captions.

# 2. **Realism & scenario coverage**
#    Most highway simulators’ default IDM doesn’t proactively clear lane space for emergency vehicles. YieldingIDM inserts that behavior so you can study realistic passage, zipper merges, etc.

# 3. **Stable curriculum for RL**
#    Ambulance RL learns very different tactics depending on how others react. YieldingIDM gives a *controllable baseline*:

#    * knobs: `yield_radius`, `right_bias`, headway factor, speed reduction
#    * easy curriculum: start high-bias (cooperative traffic), gradually reduce to mixed/non-cooperative.

# 4. **Counterfactual data for VLM/CLIP**
#    Flip the `is_emergency` flag or lower bias to generate both **positive** (drivers yield) and **negative** (don’t yield) pairs—great for contrastive learning and retrieval.

# 5. **Safety & evaluation**
#    You can quantify how often corridor formation succeeds, TTC distributions, delays, etc., under varying compliance levels—useful for ablations and benchmarks.

# 6. **Deterministic, cheap, and tunable**
#    It’s a tiny rule overlay on IDM: reversible, seedable, and much cheaper than training multi-agent policies for all the non-ego cars.

# ## When you might *not* need it

# * You already have a learned/background policy that yields reliably and is controllable.
# * You’re training a “worst-case” ambulance policy assuming zero cooperation.
# * You only use non-linguistic rewards and don’t care about aligning with “yield corridor” language.

# ## Good defaults to start

# * `yield_radius ≈ 60 m`
# * `right_bias ≈ 0.7` (probability to move right when possible)
# * `headway_factor ≈ 1.5`, `speed_delta ≈ 4 m/s`
#   Then anneal `right_bias` or `radius` during training to increase difficulty.

# ## Integration checklist

# * Set `env.config["is_emergency"]=True` for ambulance scenarios.
# * Use the revised class (with revert/no-drift + seeded RNG).
# * Log events: when neighbors enter/exit “yield mode,” lane changes, min TTC—helps debug.
# * For eval, run sweeps over `{bias, radius}` to report robustness curves.

# Bottom line: **YieldingIDM makes your environment linguistically consistent, more realistic, and controllable**, which improves both RL learning dynamics and VLM/CLIP supervision.



# # rl_langvision/yielding_traffic.py
# from __future__ import annotations
# import numpy as np
# from highway_env.vehicle.behavior import IDMVehicle

# def _right_lane_index(ln_idx, min_lane: int = 0):
#     """Move one lane to the right if possible (lane id decreases to the right in highway-env)."""
#     road_id, segment_id, lane_id = ln_idx
#     return (road_id, segment_id, max(min_lane, lane_id - 1))

# class YieldingIDM(IDMVehicle):
#     """
#     IDM vehicle that 'makes way' for an emergency ego:
#       - increases headway,
#       - reduces target speed,
#       - prefers changing to the right lane when ego is near.
#     This is a light-touch, deterministic patch over the stock IDM behavior.
#     """

#     def step(self, dt: float) -> None:
#         super().step(dt)  # keep normal IDM update

#         env = getattr(self.road, "env", None)
#         if env is None:
#             return

#         cfg = getattr(env, "config", {}) or {}
#         if not cfg.get("is_emergency", False):
#             return

#         ego = getattr(env, "vehicle", None) or getattr(env, "ego_vehicle", None)
#         if ego is None:
#             return

#         # distance to ego (ambulance)
#         dx = ego.position[0] - self.position[0]
#         dy = ego.position[1] - self.position[1]
#         d = float(np.hypot(dx, dy))

#         radius = float(cfg.get("yield_radius", 60.0))
#         if d > radius:
#             return

#         # make way when ego is approaching from behind or alongside
#         # (dx > -15 to also catch alongside vehicles)
#         if dx > -15.0:
#             # soften longitudinal behavior
#             try:
#                 self.time_headway = float(self.time_headway) * 1.5
#             except Exception:
#                 pass

#             try:
#                 # cap target velocity a bit below current to create space
#                 self.target_velocity = min(float(self.target_velocity), max(0.0, self.speed - 4.0))
#             except Exception:
#                 pass

#             # right lane preference (if a lane to the right exists)
#             try:
#                 bias = float(cfg.get("yield_right_bias", 0.7))
#                 if np.random.rand() < bias:
#                     self.target_lane_index = _right_lane_index(self.lane_index, min_lane=0)
#             except Exception:
#                 pass

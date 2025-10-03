# Corner and Roundabout Visualization Guide

## Quick Start - Watch Corner and Circle Scenarios in Action! 🚗⭕

This guide shows you how to visualize **corner/intersection** and **roundabout/circle** scenarios using real highway-env rendering (like Farama documentation).

---

## 🎯 What You'll See

### Available Scenarios:

**🌐 Corner/Intersection Scenarios (5 total):**
1. `corner_sharp_turn` - Sharp corner with oncoming traffic
2. `intersection_t_junction` - T-intersection requiring traffic to yield
3. `intersection_four_way` - Busy 4-way intersection
4. `corner_blind_curve` - Blind corner with limited visibility
5. `corner_urban_crossing` - Urban corner with complex traffic

**⭕ Roundabout/Circle Scenarios (3 total):**
1. `roundabout_single_lane` - Single-lane roundabout with yielding traffic
2. `roundabout_multi_lane` - Busy multi-lane roundabout
3. `roundabout_congested` - Congested roundabout requiring assertive navigation

---

## 🚀 How to Run

### Step 1: Activate Virtual Environment
```bash
cd /home/chettra/ITC/Research/AVs
source avs_venv/bin/activate
```

### Step 2: Run Visualization

**List all scenarios:**
```bash
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py --list
```

**Watch a specific roundabout:**
```bash
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario roundabout_single_lane \
    --duration 300 \
    --speed 1.0
```

**Watch a specific corner/intersection:**
```bash
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario intersection_four_way \
    --duration 300 \
    --speed 0.8
```

**Demo ALL corner and circle scenarios (8 total):**
```bash
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --demo \
    --duration 150 \
    --speed 1.5
```

---

## 🎮 Command Parameters

- `--scenario NAME` - View a specific scenario (see list above)
- `--duration STEPS` - How many steps to run (default: 500)
  - 100 steps ≈ 10 seconds
  - 300 steps ≈ 30 seconds
  - 500 steps ≈ 50 seconds
- `--speed MULTIPLIER` - Simulation speed
  - `0.5` = Half speed (slow motion)
  - `1.0` = Normal speed (default)
  - `1.5` = 1.5x faster
  - `2.0` = 2x faster
- `--list` - Show all available corner and circle scenarios
- `--demo` - Run all 8 scenarios sequentially

---

## 👀 What to Look For

### Road Geometries (IMPORTANT - Different from Highway!):
- **⭕ Roundabout scenarios** = Circular road with vehicles going around the circle
- **🌐 Intersection scenarios** = Roads crossing with traffic from 4 directions
- **🔄 Corner scenarios** = Intersection with sharp turns
- **🛣️ Merge scenarios** = Highway with merging lanes

### Visual Indicators:
- **🔴 Red vehicle at top** = Ambulance (Agent 0)
- **⚪ White vehicles** = NPCs that should yield
- **🟦 Blue vehicles** = Other controlled agents (if any)

### NPC Yielding Behavior:
1. **Speed Reduction**: NPCs slow down when ambulance approaches
2. **Distance Keeping**: NPCs maintain larger following distances
3. **Lane Changes**: NPCs may change lanes to let ambulance pass
4. **Hesitation**: NPCs pause at intersections/roundabouts

### Statistics Displayed:
- Step count and progress
- Average ambulance speed
- Average NPC speed
- Yielding events detected
- Speed difference (should be 3-10 m/s)

---

## 📊 Example Outputs

**Roundabout Scenario (roundabout_single_lane):**
```
Step 150/300 | Ambulance: 24.5 m/s | NPCs: 18.2 m/s | Yielding Events: 47
```
✅ NPCs ARE YIELDING! (Ambulance is 6.3 m/s faster)

**Corner Scenario (intersection_four_way):**
```
Step 150/300 | Ambulance: 22.1 m/s | NPCs: 16.8 m/s | Yielding Events: 32
```
✅ NPCs ARE YIELDING! (Ambulance is 5.3 m/s faster)

---

## 🎬 Quick Examples

### Example 1: Quick Look at Roundabout
```bash
cd /home/chettra/ITC/Research/AVs
source avs_venv/bin/activate

python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario roundabout_single_lane \
    --duration 200 \
    --speed 1.0
```

### Example 2: Slow Motion Corner Analysis
```bash
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario corner_sharp_turn \
    --duration 300 \
    --speed 0.5
```

### Example 3: Fast Demo of All Scenarios
```bash
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --demo \
    --duration 100 \
    --speed 2.0
```

---

## 🔍 Verifying Data Collection

These visualizations prove that:

1. ✅ **Corner/Intersection scenarios work** - You can see them in action
2. ✅ **Roundabout/Circle scenarios work** - Real-time highway-env rendering
3. ✅ **NPCs actually yield** - Speed differences and behavior visible
4. ✅ **Scenarios are diverse** - 8 different scenario types beyond basic highway
5. ✅ **Data collection will capture this** - What you see is what gets recorded

When you run parallel data collection with:
```bash
python collecting_ambulance_data/examples/parallel_ambulance_collection.py \
    --episodes 1000 \
    --max-steps 100 \
    --output-dir data/ambulance_dataset_30k_cpu \
    --max-workers 20 \
    --batch-optimize \
    --seed 42
```

**All 30 scenarios** (including these 8 corner/circle scenarios) will be collected!

---

## 💡 Tips

1. **Best visualization speed**: Use `--speed 0.8` to see yielding clearly
2. **Quick check**: Use `--duration 150` for fast verification
3. **Detailed analysis**: Use `--speed 0.5 --duration 500` for slow motion
4. **Full demo**: Use `--demo` to see all 8 scenarios

---

## 🎯 Quick Test - See All 3 Road Geometries

**Want to see all three road types in action?**

```bash
cd /home/chettra/ITC/Research/AVs
./collecting_ambulance_data/scenarios/test_geometries.sh
```

This will show you:
1. ⭕ **Roundabout** (circular road) - 150 steps
2. 🌐 **Intersection** (crossing roads) - 150 steps  
3. � **Corner** (sharp turn) - 150 steps

Each runs for ~15 seconds so you can clearly see the different geometries!

---

## �🐛 Troubleshooting

**If you only see highway (straight road):**
- ✅ FIXED! Script now uses:
  - `roundabout-v0` for roundabout scenarios (circular road)
  - `intersection-v0` for corner/intersection scenarios (crossing roads)
  - `merge-v0` for merge scenarios (merging lanes)
  - `highway-v0` for highway scenarios (straight road)

**If pygame window doesn't appear:**
```bash
# Check if highway-env is installed
python3 -c "import highway_env; print('✅ highway-env installed')"

# Check if display is available
echo $DISPLAY
```

**If "Environment doesn't exist" error:**
- Make sure `import highway_env` is at the top of the script ✅ (already fixed)
- Virtual environment must be activated

**If script runs but no window:**
- You may be on a headless system
- Try with X11 forwarding or VNC

---

## 📝 Summary

You now have 8 additional scenarios (beyond the original 15 highway scenarios):
- **5 Corner/Intersection scenarios** with complex traffic patterns
- **3 Roundabout/Circle scenarios** with yielding behavior

All can be visualized in real-time using highway-env's native rendering, just like in the Farama documentation! 🎉

**Next Step**: Run parallel data collection to capture all 30 scenarios!

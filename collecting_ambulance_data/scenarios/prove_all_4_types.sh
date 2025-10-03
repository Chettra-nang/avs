#!/bin/bash
# Quick Visual Proof: Run one scenario from each of the 4 environment types

cd /home/chettra/ITC/Research/AVs
source avs_venv/bin/activate

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🎬 VISUAL PROOF: All 4 Environment Types"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This will show you 4 different road geometries:"
echo "  1. 🛣️  Highway (straight roads)"
echo "  2. ⭕ Roundabout (circular roads) ← THIS IS THE KEY ONE!"
echo "  3. 🌐 Intersection (crossing roads)"
echo "  4. 🔀 Merge (merging lanes)"
echo ""
echo "Each runs for 15 seconds. Watch the road shape!"
echo ""
read -p "Press Enter to start..."

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "1/4: HIGHWAY (should see straight multi-lane road)"
echo "────────────────────────────────────────────────────────────────────────────────"
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario highway_rush_hour \
    --duration 150 \
    --speed 1.0

sleep 2

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "2/4: ROUNDABOUT (should see CIRCULAR road) ⭕"
echo "────────────────────────────────────────────────────────────────────────────────"
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario roundabout_single_lane \
    --duration 150 \
    --speed 1.0

sleep 2

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "3/4: INTERSECTION (should see crossing roads) 🌐"
echo "────────────────────────────────────────────────────────────────────────────────"
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario intersection_four_way \
    --duration 150 \
    --speed 1.0

sleep 2

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "4/4: MERGE (should see merging lanes) 🔀"
echo "────────────────────────────────────────────────────────────────────────────────"
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario merge_highway_entry \
    --duration 150 \
    --speed 1.0

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ PROOF COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "You just saw 4 different road geometries:"
echo "  ✅ Straight highway"
echo "  ✅ Circular roundabout ⭕"
echo "  ✅ Crossing intersection 🌐"
echo "  ✅ Merging lanes 🔀"
echo ""
echo "Your 30 scenarios include:"
echo "  • 13 highway scenarios (straight roads)"
echo "  • 3 roundabout scenarios (circular roads)"
echo "  • 5 intersection/corner scenarios (crossing roads)"
echo "  • 9 merge scenarios (merging lanes)"
echo ""
echo "Data collection will capture all this diversity! 🎉"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

deactivate

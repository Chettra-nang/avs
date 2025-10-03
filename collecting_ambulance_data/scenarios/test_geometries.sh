#!/bin/bash
# Test all three environment types to verify they show different road geometries

cd /home/chettra/ITC/Research/AVs
source avs_venv/bin/activate

echo "═══════════════════════════════════════════════════════════════"
echo "🎬 Testing Corner & Circle Visualizations"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "This will show 3 different road geometries:"
echo "  1. ⭕ Roundabout (circular road)"
echo "  2. 🌐 Intersection (crossing roads)"
echo "  3. 🛣️  Merge (highway merge lanes)"
echo ""
echo "Each will run for 150 steps (~15 seconds)"
echo "Press Ctrl+C at any time to skip to the next"
echo ""
read -p "Press Enter to start..."

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "1/3: Testing ROUNDABOUT (should show circular road)"
echo "──────────────────────────────────────────────────────────────"
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario roundabout_single_lane \
    --duration 150 \
    --speed 1.0

sleep 2

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "2/3: Testing INTERSECTION (should show crossing roads)"
echo "──────────────────────────────────────────────────────────────"
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario intersection_four_way \
    --duration 150 \
    --speed 1.0

sleep 2

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "3/3: Testing CORNER (should show intersection)"
echo "──────────────────────────────────────────────────────────────"
python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
    --scenario corner_sharp_turn \
    --duration 150 \
    --speed 1.0

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Test Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "You should have seen 3 different road geometries:"
echo "  ✅ Roundabout: Circular road with vehicles going around"
echo "  ✅ Intersection: Roads crossing with traffic from 4 directions"
echo "  ✅ Corner: Intersection with vehicles turning"
echo ""
echo "All 30 scenarios will be collected during data collection!"
echo ""

deactivate

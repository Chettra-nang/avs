#!/bin/bash
# Quick commands to visualize corner and circle scenarios

echo "═══════════════════════════════════════════════════════════════"
echo "🚗 Corner & Circle Scenario Visualization - Quick Commands"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Activate venv
cd /home/chettra/ITC/Research/AVs
source avs_venv/bin/activate

echo "Choose an option:"
echo ""
echo "1. List all scenarios"
echo "2. Watch roundabout_single_lane (recommended)"
echo "3. Watch intersection_four_way"
echo "4. Watch corner_sharp_turn"
echo "5. Demo all 8 scenarios (quick)"
echo "6. Demo all 8 scenarios (detailed)"
echo ""
read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo "📋 Listing all corner and circle scenarios..."
        python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py --list
        ;;
    2)
        echo "⭕ Watching roundabout_single_lane..."
        python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
            --scenario roundabout_single_lane \
            --duration 300 \
            --speed 1.0
        ;;
    3)
        echo "🌐 Watching intersection_four_way..."
        python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
            --scenario intersection_four_way \
            --duration 300 \
            --speed 0.8
        ;;
    4)
        echo "🔄 Watching corner_sharp_turn..."
        python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
            --scenario corner_sharp_turn \
            --duration 300 \
            --speed 0.8
        ;;
    5)
        echo "🎬 Quick demo of all 8 scenarios..."
        python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
            --demo \
            --duration 100 \
            --speed 2.0
        ;;
    6)
        echo "🎬 Detailed demo of all 8 scenarios..."
        python3 collecting_ambulance_data/scenarios/watch_corner_and_circle.py \
            --demo \
            --duration 200 \
            --speed 1.0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"
deactivate

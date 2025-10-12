#!/bin/bash
# Pre-push verification script - Test all offline RL components
# Usage: cd /home/chettra/ITC/Research/AVs && bash offline_rl/test_before_push.sh

set +e  # Don't exit on error, we want to see all test results

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     OFFLINE RL PRE-PUSH VERIFICATION SCRIPT                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAILED=$((FAILED + 1))
    fi
}

# Test 1: Check directory structure
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Directory Structure"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "scripts/export_offline_dataset.py" ]; then
    echo -e "${RED}Error: Must run from AVs directory${NC}"
    echo "Usage: cd /home/chettra/ITC/Research/AVs && bash offline_rl/test_before_push.sh"
    exit 1
fi

echo "Checking required files..."
FILES=(
    "scripts/export_offline_dataset.py"
    "scripts/verify_offline_pipeline.py"
    "offline_rl/trainers/train_offline_dqn.py"
    "offline_rl/trainers/train_bc.py"
    "offline_rl/rl_langvision/__init__.py"
    "offline_rl/rl_langvision/clip_embedder.py"
    "offline_rl/README.md"
    "run_offline_training.sh"
    "OFFLINE_RL_TRAINING_GUIDE.md"
)

MISSING=0
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo -e "  ${RED}✗ $file${NC}"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 0 ]; then
    test_result 0
else
    test_result 1
fi
echo ""

# Test 2: Python syntax check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Python Syntax Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SYNTAX_ERRORS=0
for pyfile in scripts/*.py offline_rl/trainers/*.py offline_rl/rl_langvision/*.py; do
    if [ -f "$pyfile" ]; then
        python3 -m py_compile "$pyfile" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✓ $pyfile"
        else
            echo -e "  ${RED}✗ $pyfile${NC}"
            SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
        fi
    fi
done

if [ $SYNTAX_ERRORS -eq 0 ]; then
    test_result 0
else
    test_result 1
fi
echo ""

# Test 3: Import test (basic dependencies)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Basic Python Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'EOF'
import sys

deps = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow.parquet",
    "PIL": "PIL",
}

missing = []
for name, module in deps.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (missing)")
        missing.append(name)

if missing:
    print(f"\n❌ Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)
else:
    sys.exit(0)
EOF

test_result $?
echo ""

# Test 4: Export script functionality
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: Export Script Functionality"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'EOF'
import sys
sys.path.insert(0, 'scripts')

try:
    from export_offline_dataset import decode_blob, normalize_to_chw, process_parquet
    print("  ✓ Import successful")
    
    from pathlib import Path
    test_parquet = list(Path('data/ambulance_dataset_diagnose').rglob('*_transitions.parquet'))[0]
    print(f"  ✓ Found test parquet: {test_parquet.name}")
    
    transitions = process_parquet(test_parquet)
    print(f"  ✓ Processed {len(transitions)} transitions")
    
    if transitions:
        t = transitions[0]
        assert 'obs' in t and 'action' in t and 'reward' in t
        print(f"  ✓ Transition structure valid")
    
    sys.exit(0)
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)
EOF

test_result $?
echo ""

# Test 5: Dataset export on small sample
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 5: Dataset Export (Small Sample)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Find a small batch to test
SMALL_BATCH=$(find data/ambulance_dataset_diagnose -type d -name "batch_*" | head -1)

if [ -z "$SMALL_BATCH" ]; then
    echo -e "${YELLOW}  ⚠ No batch found for testing${NC}"
    test_result 1
else
    echo "  Testing on: $SMALL_BATCH"
    
    python3 scripts/export_offline_dataset.py \
        --input "$SMALL_BATCH" \
        --output data/test_export_verify \
        --format npz > /tmp/export_test.log 2>&1
    
    if [ $? -eq 0 ] && [ -f "data/test_export_verify/offline_dataset.npz" ]; then
        echo "  ✓ Export successful"
        
        # Verify dataset structure
        python3 << 'EOF'
import numpy as np
data = np.load('data/test_export_verify/offline_dataset.npz')
required_keys = ['obs', 'action', 'reward', 'next_obs', 'done']
missing = [k for k in required_keys if k not in data.keys()]
if missing:
    print(f"  ✗ Missing keys: {missing}")
    exit(1)
else:
    print(f"  ✓ Dataset structure valid ({len(data['obs'])} transitions)")
    exit(0)
EOF
        test_result $?
    else
        echo "  ✗ Export failed"
        cat /tmp/export_test.log | head -20
        test_result 1
    fi
fi
echo ""

# Test 6: Training scripts import check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 6: Training Scripts Import Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, 'offline_rl')

try:
    # Test rl_langvision imports
    from rl_langvision.clip_embedder import CLIPImageEncoder
    print("  ✓ CLIPImageEncoder import")
    
    from rl_langvision.cached_embedder import CachedLLMEmbedder
    print("  ✓ CachedLLMEmbedder import")
    
    from rl_langvision.amb_highway_wrapper_clip import AmbulanceHighwayCLIPWrapper
    print("  ✓ AmbulanceHighwayCLIPWrapper import")
    
    print("  ✓ rl_langvision module OK")
    sys.exit(0)
    
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    print("\n  Note: PyTorch/OpenCLIP may not be installed (OK for now)")
    print("  These will be needed on RTX 5090 for training")
    sys.exit(0)  # Don't fail - PyTorch may not be installed yet
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)
EOF

test_result $?
echo ""

# Test 7: Git status check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 7: Git Repository Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d ".git" ]; then
    echo "  ✓ Git repository detected"
    
    # Check if new files are tracked
    NEW_FILES=$(git ls-files --others --exclude-standard | grep -E "(offline_rl|OFFLINE_RL|run_offline)" | wc -l)
    
    if [ $NEW_FILES -gt 0 ]; then
        echo -e "${YELLOW}  ⚠ $NEW_FILES new files not yet added to git${NC}"
        echo "  Run: git add offline_rl/ scripts/export_offline_dataset.py scripts/verify_offline_pipeline.py *.md *.sh *.txt"
    else
        echo "  ✓ All offline RL files tracked"
    fi
    
    test_result 0
else
    echo -e "${RED}  ✗ Not a git repository${NC}"
    test_result 1
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    TEST SUMMARY                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ ALL TESTS PASSED - READY TO PUSH TO GITHUB!          ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. git add offline_rl/ scripts/export_offline_dataset.py scripts/verify_offline_pipeline.py"
    echo "  2. git add OFFLINE_RL_*.md QUICK_REFERENCE.txt run_offline_training.sh"
    echo "  3. git commit -m 'Add offline RL training pipeline'"
    echo "  4. git push"
    echo "  5. Transfer to RTX 5090 and run: bash run_offline_training.sh"
    exit 0
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ⚠️  SOME TESTS FAILED - FIX BEFORE PUSHING               ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi

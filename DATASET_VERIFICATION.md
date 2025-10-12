# ✅ Dataset Configuration Check

## 📊 Current Training Dataset Configuration

### ✅ YES - Using `ambulance_dataset_diagnose`

All your training scripts are correctly configured to use:
```bash
DATA_DIR="data/ambulance_dataset_diagnose"
```

---

## 📁 File Locations

### Training Scripts (All using `ambulance_dataset_diagnose`)

1. **`run_offline_training_ULTRAFAST.sh`** ✅
   ```bash
   Line 26: DATA_DIR="data/ambulance_dataset_diagnose"
   ```

2. **`run_offline_training_FAST.sh`** ✅
   ```bash
   Line 22: DATA_DIR="data/ambulance_dataset_diagnose"
   ```

3. **`run_offline_training.sh`** ✅
   ```bash
   Line 20: DATA_DIR="data/ambulance_dataset_diagnose"
   ```

---

## 🔍 Dataset Flow

### Your Training Pipeline:

```
ambulance_dataset_diagnose/           (Raw parquet data)
    └─ combined_ambulance_data.parquet
            ↓ (export_offline_dataset.py)
data/offline_dataset/                  (Exported NPZ)
    ├─ offline_dataset.npz
    └─ clip_features.npz              (Pre-computed CLIP features)
            ↓ (train_*_ultrafast.py)
checkpoints/                           (Trained models)
    ├─ bc_pretrain/best_model.pt
    ├─ offline_dqn/best_model.pt
    └─ offline_ppo/best_model.pt
```

---

## ⚠️ HOWEVER - Check Your RTX 5090 Machine!

### Important Question:

**Where is `ambulance_dataset_diagnose` on your RTX 5090 machine?**

You mentioned in your data collection command:
```bash
--output-dir data/ambulance_dataset_30k_cpu
```

But your training scripts look for:
```bash
DATA_DIR="data/ambulance_dataset_diagnose"
```

---

## 🔍 Possible Scenarios

### Scenario A: Data is at Both Locations

You might have data in both places:
```bash
data/ambulance_dataset_diagnose/      ← Training uses THIS
data/ambulance_dataset_30k_cpu/       ← Collection saved HERE
```

**Check which one has the actual training data:**

```bash
# On RTX 5090
cd /home/admin-ubuntu/Research/avs/avs

# Check both directories
ls -lh data/ambulance_dataset_diagnose/
ls -lh data/ambulance_dataset_30k_cpu/

# Check which has the parquet file
ls -lh data/ambulance_dataset_diagnose/*.parquet
ls -lh data/ambulance_dataset_30k_cpu/*.parquet
```

---

### Scenario B: You Moved/Renamed the Data

You might have:
1. Collected to `ambulance_dataset_30k_cpu/`
2. Then moved/renamed to `ambulance_dataset_diagnose/`
3. Training is using the renamed version

**Verify:**
```bash
# Check file timestamps
stat data/ambulance_dataset_diagnose/combined_ambulance_data.parquet
stat data/ambulance_dataset_30k_cpu/combined_ambulance_data.parquet
```

---

### Scenario C: Different Datasets

You might have TWO different datasets:
- `ambulance_dataset_diagnose/` - Older/different data (used for training)
- `ambulance_dataset_30k_cpu/` - Newer data (collected but not used yet)

**This would explain low performance!** Training on old data, evaluating on new scenarios!

---

## ✅ How to Verify What Data Was Used for Training

### Check 1: Dataset Size

```bash
# On RTX 5090
cd /home/admin-ubuntu/Research/avs/avs

# Check exported dataset
python3 << 'EOF'
import numpy as np

data = np.load('data/offline_dataset/offline_dataset.npz')
print(f"Dataset transitions: {len(data['observations'])}")
print(f"Expected: 450,000 (4,500 episodes × 100 steps)")
print(f"Dataset files: {list(data.keys())}")
EOF
```

### Check 2: Training Logs

Look at your training output - it should show:
```
Loading dataset from: data/ambulance_dataset_diagnose
Total episodes: X
Total transitions: Y
```

### Check 3: Compare Datasets

```bash
# On RTX 5090
cd /home/admin-ubuntu/Research/avs/avs

python3 << 'EOF'
import pandas as pd

# Check both datasets
try:
    df1 = pd.read_parquet('data/ambulance_dataset_diagnose/combined_ambulance_data.parquet')
    print(f"ambulance_dataset_diagnose:")
    print(f"  Episodes: {df1['episode'].nunique()}")
    print(f"  Transitions: {len(df1)}")
    print(f"  Columns: {list(df1.columns)[:10]}")
except:
    print("❌ ambulance_dataset_diagnose not found")

print()

try:
    df2 = pd.read_parquet('data/ambulance_dataset_30k_cpu/combined_ambulance_data.parquet')
    print(f"ambulance_dataset_30k_cpu:")
    print(f"  Episodes: {df2['episode'].nunique()}")
    print(f"  Transitions: {len(df2)}")
    print(f"  Columns: {list(df2.columns)[:10]}")
except:
    print("❌ ambulance_dataset_30k_cpu not found")
EOF
```

---

## 🎯 Summary

### ✅ Training Scripts Configuration

**All correct!** Your training scripts are configured to use:
```bash
data/ambulance_dataset_diagnose/
```

### ⚠️ But You Need to Verify

1. **Does `ambulance_dataset_diagnose` exist on RTX 5090?**
2. **Is it the same data you collected (150 episodes × 30 scenarios)?**
3. **Or is it different/older data?**

### 🔍 Run This on RTX 5090 to Check

```bash
cd /home/admin-ubuntu/Research/avs/avs

# Check what datasets exist
echo "=== Checking datasets ==="
ls -lh data/ambulance*/

# Compare dataset sizes
python3 scripts/analyze_dataset_quality.py --data-dir data/ambulance_dataset_diagnose
python3 scripts/analyze_dataset_quality.py --data-dir data/ambulance_dataset_30k_cpu
```

This will tell you:
1. Which datasets exist
2. Their sizes and quality
3. Whether training used the expected data

---

## 💡 Recommendation

**Most likely scenario**: You have the right data, just need to confirm:

1. ✅ Training scripts look for `ambulance_dataset_diagnose`
2. ✅ That directory exists and has your 4,500 episodes
3. ✅ Models were trained on that data

**If `ambulance_dataset_diagnose` is empty/missing**:
→ Either copy `ambulance_dataset_30k_cpu` to `ambulance_dataset_diagnose`
→ Or update training scripts to use `ambulance_dataset_30k_cpu`

**Report back** the output of the verification commands and I'll help you confirm! 🔍

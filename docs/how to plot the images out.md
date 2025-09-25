<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# how to plot the images out

It’s possible to reconstruct and plot the stored image stacks by decoding each row’s grayscale_blob using its shape and dtype columns, then displaying with matplotlib; below is a minimal loader and viewer for Parquet or CSV logs.[^1]

### Python setup

- Use pandas (pyarrow for Parquet), NumPy to reshape buffers, and matplotlib to render frames from the 4×H×W grayscale stacks.[^1]


### Decode and plot a sample

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Parquet (or CSV if applicable)
path = "transitions.parquet"           # replace with actual file
df = pd.read_parquet(path)             # for CSV: pd.read_csv("...csv")

# Helper to decode a binary blob to numpy array using per-row shape/dtype
def decode_blob(row, blob_col, shape_col, dtype_col):
    buf = row[blob_col]
    if pd.isna(buf):
        return None
    # shape may be stored as a Python-like string "[4, 128, 64]" or a list/tuple
    shape = row[shape_col]
    if isinstance(shape, str):
        shape = tuple(int(s) for s in shape.strip("[]()").split(",") if s.strip())
    else:
        shape = tuple(shape)
    dtype = np.dtype(row[dtype_col])
    arr = np.frombuffer(buf, dtype=dtype)
    return arr.reshape(shape)

# Pick a row to visualize
row = df.iloc[^0]
stack = decode_blob(row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
print("Decoded stack shape:", None if stack is None else stack.shape)

# Plot all frames in the stack (assumed shape: (T, H, W))
if stack is not None:
    T = stack.shape[^0]
    cols = min(T, 4)
    rows = int(np.ceil(T / cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for t in range(T):
        plt.subplot(rows, cols, t+1)
        plt.imshow(stack[t], cmap="gray", vmin=0, vmax=255 if stack.dtype==np.uint8 else None)
        plt.title(f"Frame {t}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
```


### Batch export to PNGs

```python
import os
os.makedirs("frames_out", exist_ok=True)

def save_stack(row, prefix):
    stack = decode_blob(row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
    if stack is None: 
        return
    for t in range(stack.shape[^0]):
        plt.imsave(f"frames_out/{prefix}_f{t}.png", stack[t], cmap="gray")

for i, row in df.head(5).iterrows():
    save_stack(row, f"ep{row['episode_id']}_step{row['step']}")
```


### Plot occupancy grids

```python
row = df.iloc[^0]
occ = decode_blob(row, "occupancy_blob", "occupancy_shape", "occupancy_dtype")
if occ is not None:
    # If shape like (C, H, W) or (2, 11, 11), show channels side-by-side
    C = occ.shape[^0]
    plt.figure(figsize=(3*C, 3))
    for c in range(C):
        plt.subplot(1, C, c+1)
        plt.imshow(occ[c], cmap="magma")
        plt.title(f"Occ ch{c}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
```


### Common pitfalls

- Mixed dtypes: honor grayscale_dtype; many logs use "|u1" (uint8) for images and float32 for occupancy. [^1]
- Shape parsing: if shapes are stored as strings, parse carefully to integers; if stored as lists in Parquet, they may already be lists.[^1]
- Endianness: np.frombuffer will respect dtype; ensure no extra compression at the column level (Parquet handles it transparently).[^1]


### Verifications

- After decoding, check that product(grayscale_shape) equals len(grayscale_blob) in bytes divided by itemsize, and similarly for occupancy, before plotting.[^1]

```
<div style="text-align: center">⁂</div>
```

[^1]: 20250921_151946-f35b18e8_transitions.csv


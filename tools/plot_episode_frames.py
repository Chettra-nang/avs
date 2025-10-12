"""Plot frames from stored ambulance dataset Parquet files.

This script extracts grayscale frames (if present) from transitions.parquet files
and writes the first N frames for a chosen scenario/episode to PNG files.

Usage examples (run from project root):

python tools/plot_episode_frames.py --batch data/ambulance_dataset_diagnose/batch_17220 --scenario merge_highway_entry --out /tmp/ambulance_plots --nframes 5

The script will look for files named '*_transitions.parquet' in the scenario directory and
will extract frames from the 'grayscale_blob' column using corresponding
'grayscale_shape' and 'grayscale_dtype' metadata.
"""

import argparse
import os
import pyarrow.parquet as pq
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
# Optional: CLAHE via OpenCV if available. We try to import cv2 but continue
# gracefully if it's not installed so this script remains lightweight.
try:
    import cv2

    HAVE_CV2 = True
except Exception:
    cv2 = None
    HAVE_CV2 = False
import json


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def extract_grayscale_from_parquet(parquet_path, out_dir, nframes=5):
    table = pq.read_table(parquet_path)
    cols = table.column_names
    if 'grayscale_blob' not in cols:
        print('No grayscale_blob column in', parquet_path)
        return 0

    # convert to pandas for convenience
    df = table.to_pandas()

    # find first episode id
    if df.empty:
        print('Parquet empty:', parquet_path)
        return 0

    first_episode = df['episode_id'].iloc[0]
    episode_rows = df[df['episode_id'] == first_episode]

    saved = 0
    step_idx = 0

    # Precompute episode-wide percentiles for floating point normalization so
    # early frames use the same scale as later frames. Collect all grayscale
    # blobs for the episode (up to a modest cap to avoid excessive memory use)
    # and compute the 0.5/99.5 percentiles used above.
    episode_values = None
    float_percentiles = (0.5, 99.5)
    try:
        float_blobs = []
        max_blobs = 200  # safety cap
        for _, r in episode_rows.iterrows():
            b = r.get('grayscale_blob')
            s = r.get('grayscale_shape')
            d = r.get('grayscale_dtype')
            if b is None or s is None or d is None:
                continue
            npdtype = np.dtype(d)
            arr = np.frombuffer(b, dtype=npdtype)
            expected = int(np.prod(s))
            if arr.size < expected:
                continue
            if arr.size > expected:
                arr = arr[:expected]
            try:
                arr = arr.reshape(tuple(s))
            except Exception:
                continue
            # flatten and sample to limit memory
            flat = np.ravel(arr).astype('float32')
            if flat.size > 1000000:
                # sample uniformly if very large
                idx = np.linspace(0, flat.size - 1, 1000000).astype(int)
                flat = flat[idx]
            float_blobs.append(flat)
            if len(float_blobs) >= max_blobs:
                break
        if float_blobs:
            episode_values = np.concatenate(float_blobs)
    except Exception:
        episode_values = None

    if episode_values is not None and episode_values.size > 0:
        try:
            ep_p_low, ep_p_high = np.percentile(episode_values, float_percentiles)
        except Exception:
            ep_p_low, ep_p_high = None, None
    else:
        ep_p_low, ep_p_high = None, None

    for _, row in episode_rows.iterrows():
        if saved >= nframes:
            break
        blob = row.get('grayscale_blob')
        shape = row.get('grayscale_shape')
        dtype = row.get('grayscale_dtype')
        if blob is None or shape is None or dtype is None:
            continue
        # blob is bytes; reconstruct
        # Try to load arrays written with np.save (npy format). If that fails,
        # fall back to interpreting the blob as raw dtype bytes with np.frombuffer.
        import io
        npdtype = np.dtype(dtype)
        try:
            buf = io.BytesIO(blob)
            # This will succeed if the blob was written with np.save
            arr_loaded = np.load(buf, allow_pickle=False)
            arr = np.asarray(arr_loaded)
        except Exception:
            # Fallback for raw-bytes encoding
            arr = np.frombuffer(blob, dtype=npdtype)
        expected_elems = int(np.prod(shape))
        if arr.size < expected_elems:
            print(f'Blob too small ({arr.size}) for expected shape {shape} ({expected_elems})')
            continue
        if arr.size > expected_elems:
            # Truncate any trailing padding
            arr = arr[:expected_elems]
        try:
            arr = arr.reshape(tuple(shape))
        except Exception as e:
            print('Failed to reshape grayscale blob after slicing:', e)
            continue
        # If arr has channels (H,W,C) pick single channel or convert
        # Convert float images to 0..255 uint8 for saving
        # Robust normalization to reduce influence of outliers.
        # - For floating point data use 1st/99th percentile clipping then scale to 0..255.
        # - For integer data, if values look like 0..1 scale to 0..255, otherwise cast to uint8.
        if np.issubdtype(arr.dtype, np.floating):
            # Use episode-wide percentiles when available, otherwise fall back
            # to per-frame percentiles.
            if ep_p_low is not None and ep_p_high is not None:
                p_low, p_high = float(ep_p_low), float(ep_p_high)
            else:
                p_low, p_high = np.percentile(arr, [0.5, 99.5])

            if p_high > p_low:
                arr_clipped = np.clip(arr, p_low, p_high)
                norm = (arr_clipped - p_low) / (p_high - p_low)
            else:
                # flat image
                norm = np.clip(arr - p_low if p_low is not None else arr, 0.0, 1.0)
            arr_uint8 = (np.nan_to_num(norm) * 255.0).astype(np.uint8)
        else:
            # integer types
            try:
                amax = int(arr.max())
            except Exception:
                amax = None
            # common case: floats encoded as 0/1 integers -> scale up
            if amax is not None and amax <= 1:
                arr_uint8 = (arr.astype('float32') * 255.0).astype(np.uint8)
            else:
                arr_uint8 = arr.astype(np.uint8)

        # Squeeze singleton dimensions then normalize to canonical (C,H,W)
        # to avoid ambiguous axis ordering between (C,H,W), (H,W,C) or (N,H,W).
        arr_uint8 = np.squeeze(arr_uint8)

        def normalize_to_chw(a, meta_shape=None):
            """Return a uint8 ndarray shaped (C, H, W).

            Heuristics used (in order):
            - If meta_shape given and is a permutation of a.shape, reorder to match meta_shape.
            - If 2D -> (1, H, W).
            - If 3D:
              * If first dim is small (<=4) and last two dims are unequal -> likely (C,H,W).
              * If last dim is small (<=4) -> likely (H,W,C) -> transpose.
              * If first dim > 4 and last two dims are plausible H,W -> likely (N,H,W) -> take first channel.
              * Fallback: if any dim equals 3 treat that as channel dim appropriately.
            The function is conservative and logs when heuristics are applied.
            """
            a = np.asarray(a)
            # quick metadata-assisted permutation
            if meta_shape is not None and a.ndim == len(meta_shape):
                # try to find permutation p such that tuple(a.shape[i] for i in p) == meta_shape
                from itertools import permutations

                try:
                    for p in permutations(range(a.ndim)):
                        if tuple(a.shape[i] for i in p) == tuple(int(x) for x in meta_shape):
                            # reorder
                            a = a.transpose(p)
                            return a.astype('uint8')
                except Exception:
                    pass

            if a.ndim == 2:
                H, W = a.shape
                return a.reshape((1, H, W)).astype('uint8')

            if a.ndim == 3:
                d0, d1, d2 = a.shape
                # common channel counts
                channel_candidates = {1, 3, 4}
                if d0 in channel_candidates and d1 != d2:
                    # (C,H,W)
                    return a.astype('uint8')
                if d2 in channel_candidates:
                    # (H,W,C)
                    return a.transpose(2, 0, 1).astype('uint8')
                # If leading dimension larger than typical channel counts, assume (N,H,W)
                if d0 > 4 and d1 != d2:
                    # take first frame/channel
                    return a[0].reshape((1, a[0].shape[0], a[0].shape[1])).astype('uint8')
                # Fallback heuristics: if any dim==3 assume that is channels
                if d0 == 3:
                    return a.astype('uint8')
                if d1 == 3:
                    return a.transpose(1, 2, 0).transpose(2, 0, 1).astype('uint8')
                if d2 == 3:
                    return a.transpose(2, 0, 1).astype('uint8')

            # If we can't decide, flatten to (1,H,W) using last two dims as H,W
            try:
                H, W = a.shape[-2], a.shape[-1]
                return a.reshape((1, H, W)).astype('uint8')
            except Exception:
                return a.astype('uint8')

        chw = normalize_to_chw(arr_uint8, meta_shape=shape)
        # create PIL images from canonical (C,H,W)
        # stacked_rgb: the original behavior (stack up to first 3 channels)
        stacked_img = None
        last_frame_img = None
        if chw.ndim == 3:
            C, H, W = chw.shape
            if C == 1:
                stacked_img = Image.fromarray(chw[0])
                last_frame_img = Image.fromarray(chw[0])
            else:
                # take up to first 3 channels for RGB stacked preview (kept as secondary)
                rgb = np.stack([chw[i] if i < C else np.zeros((H, W), dtype='uint8') for i in range(3)], axis=2)
                stacked_img = Image.fromarray(rgb)
                # Default thumbnail should be the last temporal slice (most recent frame)
                last_arr = chw[-1]
                last_frame_img = Image.fromarray(last_arr)
        else:
            # fallback to 2D
            stacked_img = Image.fromarray(chw)
            last_frame_img = Image.fromarray(chw)

        # max-projection across channels: highlights any transient/bright pixels
        try:
            maxproj = np.max(chw, axis=0).astype('uint8')
            maxproj_img = Image.fromarray(maxproj)
        except Exception:
            maxproj_img = None

        # Post-process thumbnail to improve visibility: ensure grayscale and stretch.
        # Use autocontrast + equalize for a general improvement. For very-dark
        # or nearly-flat images apply a stronger brightness/contrast boost.
        def enhance_image(img_obj):
            try:
                if img_obj.mode != 'L':
                    img_obj = img_obj.convert('L')

                # Basic stretch
                img_obj = ImageOps.autocontrast(img_obj, cutoff=0)
                img_obj = ImageOps.equalize(img_obj)

                # If the image is still very dark or low-contrast, apply a stronger
                # enhancement pass. We measure mean in 0..1 range.
                arr_check = np.asarray(img_obj).astype('float32') / 255.0
                mean_val = float(np.nanmean(arr_check))
                std_val = float(np.nanstd(arr_check))

                # Tunable thresholds. Raise the mean threshold to catch fairly-dark
                # early frames that still look washed out after global equalization.
                DARK_MEAN_THRESHOLD = 0.22
                LOW_STD_THRESHOLD = 0.02
                EXTRA_BRIGHTNESS = 1.8
                EXTRA_CONTRAST = 1.5

                if mean_val < DARK_MEAN_THRESHOLD or std_val < LOW_STD_THRESHOLD:
                    # Second pass: try adaptive histogram equalization (CLAHE) when
                    # OpenCV is available — this often produces better local
                    # contrast than global equalize for low-dynamic-range blobs.
                    try:
                        if HAVE_CV2:
                            img_np = np.asarray(img_obj)
                            # ensure uint8
                            if img_np.dtype != np.uint8:
                                img_np = (np.clip(img_np, 0, 255)).astype('uint8')
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                            img_np = clahe.apply(img_np)
                            img_obj = Image.fromarray(img_np)
                        else:
                            # fallback: slightly stronger global autocontrast/equalize
                            img_obj = ImageOps.autocontrast(img_obj, cutoff=1)
                            img_obj = ImageOps.equalize(img_obj)
                            img_obj = ImageEnhance.Brightness(img_obj).enhance(EXTRA_BRIGHTNESS)
                            img_obj = ImageEnhance.Contrast(img_obj).enhance(EXTRA_CONTRAST)
                    except Exception:
                        # If CLAHE fails for any reason, fall back to the original
                        # enhancement pipeline to avoid crashing the extractor.
                        try:
                            img_obj = ImageOps.autocontrast(img_obj, cutoff=1)
                            img_obj = ImageOps.equalize(img_obj)
                            img_obj = ImageEnhance.Brightness(img_obj).enhance(EXTRA_BRIGHTNESS)
                            img_obj = ImageEnhance.Contrast(img_obj).enhance(EXTRA_CONTRAST)
                        except Exception:
                            pass
            except Exception:
                # If PIL ops fail for unexpected reasons, continue without enhancement
                return img_obj
            return img_obj

        # Enhance the default (last-frame) thumbnail and the max-projection image (if available).
        # We keep the stacked RGB image available but do not use it as the default thumbnail.
        img = enhance_image(last_frame_img)
        if maxproj_img is not None:
            maxproj_img = enhance_image(maxproj_img)

        out_path = os.path.join(out_dir, f'{os.path.basename(parquet_path)}.{first_episode}.step{step_idx}.png')
        img.save(out_path)
        print('Saved (last-frame)', out_path)
        # Also save the max-projection thumbnail to help with low-contrast frames
        if maxproj_img is not None:
            out_path_max = os.path.join(out_dir, f'{os.path.basename(parquet_path)}.{first_episode}.step{step_idx}.maxproj.png')
            try:
                maxproj_img.save(out_path_max)
                print('Saved (max-proj)', out_path_max)
            except Exception:
                pass
        saved += 1
        step_idx += 1

    return saved


def find_parquet_in_scenario(batch_dir, scenario_name):
    scen_dir = os.path.join(batch_dir, scenario_name)
    if not os.path.isdir(scen_dir):
        raise FileNotFoundError(scen_dir)
    files = os.listdir(scen_dir)
    parquet_files = [os.path.join(scen_dir, f) for f in files if f.endswith('_transitions.parquet')]
    return parquet_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True, help='Path to batch directory (e.g. data/ambulance_dataset_diagnose/batch_17220)')
    parser.add_argument('--scenario', required=True, help='Scenario folder name inside the batch (e.g. merge_highway_entry)')
    parser.add_argument('--out', required=True, help='Output directory for images')
    parser.add_argument('--nframes', type=int, default=5, help='Number of frames to extract')

    args = parser.parse_args()

    parquet_files = find_parquet_in_scenario(args.batch, args.scenario)
    if not parquet_files:
        print('No parquet files found for scenario', args.scenario)
        return

    ensure_dir(args.out)
    total_saved = 0
    for p in parquet_files:
        saved = extract_grayscale_from_parquet(p, args.out, nframes=args.nframes - total_saved)
        total_saved += saved
        if total_saved >= args.nframes:
            break

    print(f'Total frames saved: {total_saved}')


if __name__ == '__main__':
    main()

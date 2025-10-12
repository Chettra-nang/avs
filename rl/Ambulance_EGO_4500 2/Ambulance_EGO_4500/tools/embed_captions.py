# tools/embed_captions.py
# from __future__ import annotations
# import os, json, time
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# import pandas as pd

# tools/embed_captions.py
from __future__ import annotations
import os, json, time, sqlite3, hashlib, sys, re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Tuple
import numpy as np
import pandas as pd

# ===================== Config via ENV =====================
PROVIDER       = os.getenv("EMBED_PROVIDER", "local")            # local | ollama
CAPTIONS_DIR   = Path(os.getenv("CAPTIONS_DIR", "/Users/nginkimlong/Documents/PHD/Exchange Program (SEED)/Ambulance_EGO_4500/ambulance_dataset_fast_150_espisode_cpu_30_senario/captions"))
OUT_DIR        = Path(os.getenv("EMBED_OUT_DIR", "/Users/nginkimlong/Documents/PHD/Exchange Program (SEED)/Ambulance_EGO_4500/ambulance_dataset_fast_150_espisode_cpu_30_senario/cached_llm_v2"))
BATCH_SIZE     = int(os.getenv("EMBED_BATCH", "128"))
PARQUET_COMP   = os.getenv("PARQUET_COMP", "zstd")

# caption conflict strategy: keep_all | pick_first | pick_lowest_ttc | merge_concat
CAPTION_STRATEGY = os.getenv("CAPTION_STRATEGY", "keep_all").lower()

# Local ST
LOCAL_MODEL    = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ollama
OLLAMA_MODEL   = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11433")
OLLAMA_RETRY   = int(os.getenv("OLLAMA_RETRY", "3"))
OLLAMA_SLEEP   = float(os.getenv("OLLAMA_SLEEP", "0.01"))

# Cache toggle (SQLite optional)
USE_SQLITE     = os.getenv("EMBED_USE_SQLITE", "1") == "1"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== Optional SQLite cache (text -> vector) =====================
DB_PATH = OUT_DIR / "embed_cache.sqlite3"

def _db_connect() -> Optional[sqlite3.Connection]:
    if not USE_SQLITE:
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings(
            text_hash TEXT PRIMARY KEY,
            text      TEXT NOT NULL,
            dim       INTEGER NOT NULL,
            vec       BLOB NOT NULL
        );
    """)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def _hash_text(t: str) -> str:
    return hashlib.blake2b(t.encode("utf-8"), digest_size=20).hexdigest()

def cache_get_many(conn: Optional[sqlite3.Connection], uniq_hashes: List[str]) -> Dict[str, np.ndarray]:
    if not conn or not uniq_hashes:
        return {}
    out: Dict[str, np.ndarray] = {}
    CH = 800
    for i in range(0, len(uniq_hashes), CH):
        chunk = uniq_hashes[i:i+CH]
        q = f"SELECT text_hash, dim, vec FROM embeddings WHERE text_hash IN ({','.join(['?']*len(chunk))})"
        for th, dim, blob in conn.execute(q, chunk):
            arr = np.frombuffer(blob, dtype=np.float32)
            if arr.size == dim:
                out[th] = arr
    return out

def cache_put_many(conn: Optional[sqlite3.Connection], items: List[Tuple[str, str, np.ndarray, int]]):
    if not conn or not items:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings(text_hash, text, dim, vec) VALUES (?,?,?,?)",
        [(th, t, v.size, v.astype(np.float32).tobytes()) for th, t, v, _ in items]
    )
    conn.commit()

# ===================== Embedding backends =====================
_local_model = None
_local_dim: Optional[int] = None
_device = "cpu"

def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.set_float32_matmul_precision("high")
            return "mps"
    except Exception:
        pass
    return "cpu"

def _local_load():
    global _local_model, _local_dim, _device
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _device = _pick_device()
        _local_model = SentenceTransformer(LOCAL_MODEL, device=_device)
        try:
            _local_dim = int(_local_model.get_sentence_embedding_dimension())
        except Exception:
            _local_dim = int(len(_local_model.encode(["probe"], normalize_embeddings=True)[0]))
    return _local_model, _local_dim

def embed_local(texts: List[str]) -> np.ndarray:
    model, _ = _local_load()
    arr = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=BATCH_SIZE, show_progress_bar=False
    ).astype(np.float32, copy=False)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    return arr

def _ollama_post(prompt: str) -> List[float]:
    import requests
    last = None
    for attempt in range(1, OLLAMA_RETRY+1):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                              json={"model": OLLAMA_MODEL, "prompt": prompt}, timeout=120)
            r.raise_for_status()
            return r.json()["embedding"]
        except Exception as e:
            last = e
            time.sleep(0.3 * attempt)
    raise RuntimeError(f"Ollama embedding failed after {OLLAMA_RETRY} tries: {last}")

def embed_ollama(texts: List[str]) -> np.ndarray:
    out = []
    for t in texts:
        out.append(_ollama_post(t))
        time.sleep(OLLAMA_SLEEP)
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / norms).astype(np.float32, copy=False)

def current_dim() -> int:
    if PROVIDER == "ollama":
        return int(embed_ollama(["probe"]).shape[1])
    _, d = _local_load()
    return int(d)

# ===================== Core: embed with *internal* de-dup + optional cache =====================
def embed_texts(conn: Optional[sqlite3.Connection], texts: List[str]) -> List[np.ndarray]:
    """Embeds a list of texts returning a vector per input, with internal de-dup."""
    if not texts:
        return []
    # 1) de-dup inputs
    arr_obj = np.array(texts, dtype=object)
    uniq_texts, inv_idx = np.unique(arr_obj, return_inverse=True)
    uniq_texts = uniq_texts.tolist()
    uniq_hashes = [_hash_text(t) for t in uniq_texts]

    # 2) get cached vecs for unique texts
    cached = cache_get_many(conn, uniq_hashes)
    vec_map: Dict[str, np.ndarray] = dict(cached)

    # 3) embed misses (unique)
    misses = [(t, h) for t, h in zip(uniq_texts, uniq_hashes) if h not in vec_map]
    for j in range(0, len(misses), BATCH_SIZE):
        chunk = misses[j:j+BATCH_SIZE]
        if not chunk: continue
        chunk_texts = [t for t, _ in chunk]
        arr = embed_ollama(chunk_texts) if PROVIDER == "ollama" else embed_local(chunk_texts)
        for k, (t, h) in enumerate(chunk):
            vec_map[h] = arr[k]
        cache_put_many(conn, [(h, t, vec_map[h], vec_map[h].shape[0]) for (t, h) in chunk])

    # 4) rebuild in original order
    uniq_vecs = [vec_map[h] for h in uniq_hashes]
    all_vecs = np.vstack(uniq_vecs)[inv_idx]
    return [v for v in all_vecs]

# ===================== JSONL reading =====================
def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

# ===== Caption strategy helpers =====
_TTC_RE = re.compile(r"ttc=([\d\.]+)s", re.I)

def _ttc_of(text: str) -> float:
    m = _TTC_RE.search(text)
    return float(m.group(1)) if m else float("inf")

def _group_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str,int], List[str]]:
    buckets: Dict[Tuple[str,int], List[str]] = {}
    for r in rows:
        ep = str(r.get("episode_id",""))
        st = int(r.get("step", 0))
        tx = str(r.get("text",""))
        buckets.setdefault((ep, st), []).append(tx)
    return buckets

def _select_texts_per_strategy(buckets: Dict[Tuple[str,int], List[str]]) -> Tuple[List[str], List[str], List[int], List[int]]:
    out_ep, out_step, out_cidx, texts = [], [], [], []
    for (ep, st), lst in buckets.items():
        if CAPTION_STRATEGY == "pick_first":
            chosen = [lst[0]] if lst else []
        elif CAPTION_STRATEGY == "pick_lowest_ttc":
            chosen = [min(lst, key=_ttc_of)]
        elif CAPTION_STRATEGY == "merge_concat":
            chosen = [" || ".join(lst)]
        else:  # keep_all (default)
            chosen = lst

        if CAPTION_STRATEGY == "keep_all":
            for k, t in enumerate(chosen):
                out_ep.append(ep); out_step.append(st); out_cidx.append(k); texts.append(t)
        else:
            out_ep.append(ep); out_step.append(st); out_cidx.append(0); texts.append(chosen[0] if chosen else "")
    return texts, out_ep, out_step, out_cidx

# ===================== Per-file processing =====================
def process_file(conn: Optional[sqlite3.Connection], path: Path) -> Optional[Path]:
    rows = list(_iter_jsonl(path))
    if not rows:
        return None

    buckets = _group_rows(rows)
    texts, eps, steps, cidx = _select_texts_per_strategy(buckets)

    vecs = embed_texts(conn, texts)
    emb_lists = [v.astype(np.float32, copy=False).tolist() for v in vecs]

    df = pd.DataFrame({
        "episode_id": eps,
        "step": steps,
        "caption_idx": cidx,     # disambiguates multi-captions
        "text": texts,
        "embedding": emb_lists,
    })

    out_path = OUT_DIR / f"{path.stem}.parquet"
    df.to_parquet(out_path, index=False, compression=PARQUET_COMP)
    return out_path

# ===================== Main =====================
def main():
    print(f"[embed] provider={PROVIDER} model={LOCAL_MODEL if PROVIDER=='local' else OLLAMA_MODEL}")
    print(f"[embed] captions_dir={CAPTIONS_DIR}")
    print(f"[embed] out={OUT_DIR}  batch={BATCH_SIZE}  sqlite={'on' if USE_SQLITE else 'off'}  strategy={CAPTION_STRATEGY}")

    files = sorted(CAPTIONS_DIR.glob("*.jsonl"))
    print(f"[embed] files={len(files)}")

    conn = _db_connect()
    wrote = 0
    for i, f in enumerate(files, 1):
        p = process_file(conn, f)
        if p: wrote += 1
        if (i % 50) == 0 or i == 1:
            print(f"  .. {i}/{len(files)} processed, wrote={wrote}")

    dim = current_dim()
    (OUT_DIR / "embed_info.json").write_text(json.dumps({
        "provider": PROVIDER,
        "dim": dim,
        "model": LOCAL_MODEL if PROVIDER == "local" else OLLAMA_MODEL,
        "device": (_device if PROVIDER == "local" else "n/a"),
        "parquet_compression": PARQUET_COMP,
        "batch_size": BATCH_SIZE,
        "sqlite": USE_SQLITE,
        "caption_strategy": CAPTION_STRATEGY
    }, ensure_ascii=False, indent=2))

    print(f"[ok] done: wrote {wrote} parquet files; dim={dim}")

if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as e:
        if "sentence_transformers" in str(e) and PROVIDER == "local":
            sys.stderr.write(
                "Missing dependency: sentence-transformers.\n"
                "Install with:\n  python -m pip install sentence-transformers\n"
            )
        raise





# # === No-key local by default ===
# PROVIDER = os.getenv("EMBED_PROVIDER", "local")   # local | ollama
# CAPTIONS_DIR = Path(os.getenv("CAPTIONS_DIR", "/Users/nginkimlong/Documents/PHD/Exchange Program (SEED)/Ambulance_EGO/ambulance_dataset_15k_cpu/captions"))
# OUT_DIR      = Path(os.getenv("EMBED_OUT_DIR", "/Users/nginkimlong/Documents/PHD/Exchange Program (SEED)/Ambulance_EGO/ambulance_dataset_15k_cpu/cached_llm"))
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # -------- Local (Sentence-Transformers) backend --------
# LOCAL_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# _local_model = None
# _local_dim = None

# def _local_load():
#     global _local_model, _local_dim
#     if _local_model is None:
#         from sentence_transformers import SentenceTransformer
#         _local_model = SentenceTransformer(LOCAL_MODEL)
#         # auto-detect output dim
#         try:
#             _local_dim = _local_model.get_sentence_embedding_dimension()
#         except Exception:
#             # fallback if model doesn't expose it
#             _local_dim = len(_local_model.encode(["test"], normalize_embeddings=True)[0])
#     return _local_model, _local_dim

# def embed_local(texts: List[str]) -> List[List[float]]:
#     model, _ = _local_load()
#     vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=128)
#     return [v.tolist() for v in vecs]

# def dim_local() -> int:
#     _, d = _local_load()
#     return int(d)

# # -------- Ollama backend (optional, also no key) --------
# # Requires: `brew install ollama` ; `ollama serve` ; `ollama pull nomic-embed-text`
# OLLAMA_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")

# def embed_ollama(texts: List[str]) -> List[List[float]]:
#     import requests
#     out = []
#     for t in texts:
#         r = requests.post(
#             f"{OLLAMA_URL}/api/embeddings",
#             json={"model": OLLAMA_MODEL, "prompt": t},
#             timeout=120,
#         )
#         r.raise_for_status()
#         out.append(r.json()["embedding"])
#         time.sleep(0.01)
#     return out

# def dim_ollama() -> int:
#     # Try one quick probe (safe & cached by caller)
#     vec = embed_ollama(["probe"])[0]
#     return len(vec)

# # -------- Router --------
# _dim_cache = None

# def embed_batch(texts: List[str]) -> List[List[float]]:
#     if PROVIDER == "ollama":
#         return embed_ollama(texts)
#     return embed_local(texts)

# def current_dim() -> int:
#     global _dim_cache
#     if _dim_cache is not None:
#         return _dim_cache
#     if PROVIDER == "ollama":
#         _dim_cache = dim_ollama()
#     else:
#         _dim_cache = dim_local()
#     return _dim_cache

# # -------- File processing --------
# def process_file(path: Path) -> Optional[Path]:
#     rows = [json.loads(l) for l in path.read_text().splitlines()]
#     if not rows:
#         return None
#     texts = [r["text"] for r in rows]

#     embs: List[List[float]] = []
#     B = 128
#     for i in range(0, len(texts), B):
#         chunk = texts[i:i+B]
#         embs.extend(embed_batch(chunk))

#     df = pd.DataFrame({
#         "episode_id": [r["episode_id"] for r in rows],
#         "step": [r["step"] for r in rows],
#         "text": texts,
#         "embedding": embs,
#     })
#     out_path = OUT_DIR / f"{path.stem}.parquet"
#     df.to_parquet(out_path, index=False)
#     return out_path

# def main():
#     files = sorted(CAPTIONS_DIR.glob("*.jsonl"))
#     print(f"[embed] provider={PROVIDER} model={LOCAL_MODEL if PROVIDER=='local' else OLLAMA_MODEL}")
#     print(f"[embed] files={len(files)}  out={OUT_DIR}")

#     wrote = 0
#     for i, f in enumerate(files, 1):
#         p = process_file(f)
#         if p: wrote += 1
#         if i % 100 == 0:
#             print(f"  .. {i}/{len(files)} episodes, wrote={wrote}")

#     dim = current_dim()
#     (OUT_DIR / "embed_info.json").write_text(json.dumps({
#         "provider": PROVIDER,
#         "dim": dim,
#         "model": LOCAL_MODEL if PROVIDER=="local" else OLLAMA_MODEL
#     }))
#     print(f"[ok] done: wrote {wrote} episode-embedding parquet files; dim={dim}")

# if __name__ == "__main__":
#     main()

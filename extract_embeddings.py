# =============================================================================
# extract_embeddings.py — Phase 3 Step 1: Backbone Embedding Extraction
# =============================================================================
# Extract frozen spatio-temporal embeddings from V-JEPA 2 backbone for all
# 1010 UCF-101 videos processed in Phase 2. Save to disk for Step 2 (viz).
# =============================================================================

# =============================================================================
# Section 1 — Imports and Config
# =============================================================================
import collections
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vjepa2_webcam import model, processor, device
from vjepa2_ucf101_eval import load_frames

# Top-level constants
RESULTS_DIR     = Path("results")
EMBEDDINGS_DIR  = Path("results/embeddings")
PREDICTIONS_CSV = RESULTS_DIR / "ucf101_predictions.csv"
PATTERN_JSON    = RESULTS_DIR / "pattern_report.json"
EMBEDDINGS_NPY  = EMBEDDINGS_DIR / "embeddings.npy"
METADATA_JSON   = EMBEDDINGS_DIR / "metadata.json"
NUM_FRAMES      = 16

# =============================================================================
# Section 2 — Print Model Architecture
# =============================================================================
# Expected: Vjepa2Model with ViT-L encoder. Hook will be attached to model.vjepa2.
# The backbone output is BaseModelOutput; index [0] is last_hidden_state
# of shape (1, num_patches, hidden_dim).
print("\n=== model.vjepa2 (V-JEPA 2 backbone) ===")
print(model.vjepa2)
print(f"\nTotal backbone parameters : {sum(p.numel() for p in model.vjepa2.parameters()):,}")

try:
    hidden_dim = model.vjepa2.config.hidden_size
except AttributeError:
    hidden_dim = None  # will be inferred from first embedding extracted
print(f"Hidden dimension          : {hidden_dim}")
print("=" * 44 + "\n")


# =============================================================================
# Section 3 — Embedding Extractor
# =============================================================================
def extract_embedding(frames: np.ndarray) -> np.ndarray | None:
    """
    Extract the backbone embedding for a single video clip.

    Args:
        frames: numpy array of shape (16, H, W, 3) uint8 RGB

    Returns:
        float32 numpy array of shape (hidden_dim,), or None on error.
    """
    embedding_output: dict = {}
    hook = None

    def hook_fn(module, input, output):
        # output is BaseModelOutput or plain tensor depending on transformers version.
        # Fix vs prompt spec: use output[0] instead of output.detach() directly,
        # since BaseModelOutput is not a tensor and has no .detach() method.
        if isinstance(output, torch.Tensor):
            raw = output
        else:
            raw = output[0]  # last_hidden_state: (1, num_patches, hidden_dim)
        embedding_output["embedding"] = raw.detach().cpu()

    try:
        hook = model.vjepa2.register_forward_hook(hook_fn)

        clip_array = frames.astype(np.uint8)  # (16, H, W, 3)
        inputs = processor(videos=[clip_array], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = model(**inputs)

        raw = embedding_output["embedding"]   # (1, num_patches, hidden_dim)
        # Mean pool across patch dimension → (hidden_dim,)
        embedding_vector = raw.mean(dim=1).squeeze(0).numpy().astype(np.float32)
        return embedding_vector

    except Exception as e:
        print(f"[ERROR] extract_embedding failed: {e}")
        return None
    finally:
        if hook is not None:
            hook.remove()


# =============================================================================
# Section 4 — Batch Extraction Loop
# =============================================================================
def run_extraction(predictions_csv: Path, pattern_json: Path) -> None:
    df = pd.read_csv(predictions_csv)

    with open(pattern_json) as f:
        pattern_list = json.load(f)
    # Fix vs prompt spec: pattern_report.json is a list, not a dict.
    # Convert to dict keyed by ucf_class for O(1) lookup.
    behavior_map: dict[str, str] = {row["ucf_class"]: row["behavior"] for row in pattern_list}

    total = len(df)
    print(f"Total videos to process: {total}")

    embeddings_list: list[np.ndarray] = []
    metadata_list:   list[dict]       = []
    error_count  = 0
    embedding_idx = 0
    t0 = time.time()

    partial_npy  = EMBEDDINGS_DIR / "embeddings_partial.npy"
    partial_json = EMBEDDINGS_DIR / "metadata_partial.json"

    for _, row in tqdm(df.iterrows(), total=total, desc="Extracting"):
        # Fix vs prompt spec: video_path column holds a string; wrap in Path().
        video_path = Path(row["video_path"])
        ucf_class  = row["ucf_class"]

        frames = load_frames(video_path)
        if frames is None:
            error_count += 1
            continue

        emb = extract_embedding(frames)
        if emb is None:
            error_count += 1
            continue

        embeddings_list.append(emb)
        metadata_list.append({
            "video_path":    str(video_path),
            "ucf_class":     ucf_class,
            "behavior":      behavior_map.get(ucf_class, "UNKNOWN"),
            "top1_label":    row["top1_label"],
            "top1_prob":     float(row["top1_prob"]),
            "top1_entropy":  float(row["top1_entropy"]),
            "embedding_idx": embedding_idx,
        })
        embedding_idx += 1

        # Incremental save every 100 videos
        if embedding_idx % 100 == 0:
            np.save(partial_npy, np.array(embeddings_list, dtype=np.float32))
            with open(partial_json, "w") as f:
                json.dump(metadata_list, f)
            elapsed = time.time() - t0
            print(f"  [{embedding_idx}/{total}] embeddings saved  "
                  f"errors={error_count}  elapsed={elapsed:.1f}s")

    # Final save
    embeddings_arr = np.array(embeddings_list, dtype=np.float32)
    np.save(EMBEDDINGS_NPY, embeddings_arr)
    with open(METADATA_JSON, "w") as f:
        json.dump(metadata_list, f, indent=2)

    # Clean up partial files
    for p in (partial_npy, partial_json):
        if p.exists():
            p.unlink()

    elapsed = time.time() - t0
    print(f"\n=== Extraction Complete ===")
    print(f"  Total embeddings : {len(embeddings_list)}")
    print(f"  Embedding shape  : {embeddings_arr.shape}")
    print(f"  Embedding dim    : {embeddings_arr.shape[1] if embeddings_arr.ndim > 1 else 'N/A'}")
    print(f"  Total errors     : {error_count}")
    print(f"  Total time       : {elapsed:.1f}s")
    print(f"  embeddings.npy   : {EMBEDDINGS_NPY.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  metadata.json    : {METADATA_JSON.stat().st_size / 1024:.1f} KB")


# =============================================================================
# Section 5 — Verification
# =============================================================================
def verify_embeddings() -> None:
    embeddings = np.load(EMBEDDINGS_NPY)
    with open(METADATA_JSON) as f:
        metadata = json.load(f)

    print(f"\n=== Verification ===")
    print(f"  Embeddings array shape : {embeddings.shape}")

    # Count per behavior category
    behavior_counts = collections.Counter(m["behavior"] for m in metadata)
    print(f"\n  Embeddings per behavior:")
    for b, n in behavior_counts.most_common():
        print(f"    {b}: {n}")

    # Count per UCF class (first 10)
    class_counts = collections.Counter(m["ucf_class"] for m in metadata)
    print(f"\n  Embeddings per UCF class (first 10 by count):")
    for cls, n in list(class_counts.most_common())[:10]:
        print(f"    {cls}: {n}")

    # Mean L2 norm
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n  Mean L2 norm : {norms.mean():.4f}  (std={norms.std():.4f})")

    # Cosine similarity: same-class vs different-class
    rng = np.random.default_rng(42)

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    class_to_indices: dict[str, list[int]] = collections.defaultdict(list)
    for m in metadata:
        class_to_indices[m["ucf_class"]].append(m["embedding_idx"])

    # 5 same-class pairs
    same_sims: list[float] = []
    classes_with_pairs = [c for c, idxs in class_to_indices.items() if len(idxs) >= 2]
    chosen = rng.choice(classes_with_pairs, size=min(5, len(classes_with_pairs)), replace=False)
    for cls in chosen:
        idxs = class_to_indices[cls]
        i, j = rng.choice(idxs, size=2, replace=False)
        same_sims.append(cosine_sim(embeddings[i], embeddings[j]))

    # 5 different-class pairs
    diff_sims: list[float] = []
    all_classes = list(class_to_indices.keys())
    for _ in range(5):
        c1, c2 = rng.choice(all_classes, size=2, replace=False)
        i = rng.choice(class_to_indices[c1])
        j = rng.choice(class_to_indices[c2])
        diff_sims.append(cosine_sim(embeddings[i], embeddings[j]))

    print(f"\n  Cosine similarity — same class  (5 pairs): mean={np.mean(same_sims):.4f}")
    print(f"  Cosine similarity — diff class  (5 pairs): mean={np.mean(diff_sims):.4f}")
    print(f"  (Higher same vs diff = embeddings carry class structure)")


# =============================================================================
# Section 6 — Entry Point
# =============================================================================
if __name__ == "__main__":
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    assert PREDICTIONS_CSV.exists(), f"Run Phase 2 first: {PREDICTIONS_CSV} not found"
    assert PATTERN_JSON.exists(),    f"Run Phase 2 first: {PATTERN_JSON} not found"

    run_extraction(PREDICTIONS_CSV, PATTERN_JSON)
    verify_embeddings()
    print("Step 1 complete. Ready for Step 2 visualization.")

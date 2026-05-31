# =============================================================================
# visualize_embeddings.py — Phase 3 Step 2: Embedding Visualization
# =============================================================================
# Reduce 1024-dim V-JEPA 2 backbone embeddings to 2D via t-SNE and UMAP,
# then produce four publication-quality plot sets answering different questions
# about what the embeddings encode.
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Section 1 — Imports and Config
# =============================================================================
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import umap

# Top-level constants
EMBEDDINGS_DIR = Path("results/embeddings")
PLOTS_DIR      = Path("results/plots")
EMBEDDINGS_NPY = EMBEDDINGS_DIR / "embeddings.npy"
METADATA_JSON  = EMBEDDINGS_DIR / "metadata.json"
TSNE_SAVE      = EMBEDDINGS_DIR / "tsne_2d.npy"
UMAP_SAVE      = EMBEDDINGS_DIR / "umap_2d.npy"
RANDOM_SEED    = 42


# =============================================================================
# Section 2 — Data Loading
# =============================================================================
def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(EMBEDDINGS_NPY)
    with open(METADATA_JSON) as f:
        metadata = json.load(f)
    df = pd.DataFrame(metadata)

    assert len(embeddings) == len(df), (
        f"Shape mismatch: embeddings {len(embeddings)} vs metadata {len(df)}"
    )

    print(f"Embedding shape         : {embeddings.shape}")
    print(f"Behavior category counts:")
    for b, n in df["behavior"].value_counts().items():
        print(f"  {b}: {n}")
    print(f"Unique UCF classes      : {df['ucf_class'].nunique()}")
    print(f"Unique SSv2 labels      : {df['top1_label'].nunique()}")

    return embeddings, df


# =============================================================================
# Section 3 — Dimensionality Reduction
# =============================================================================
def compute_tsne(embeddings: np.ndarray) -> np.ndarray:
    if TSNE_SAVE.exists():
        print("  [cache] Loading t-SNE from disk.")
        return np.load(TSNE_SAVE)

    t0 = time.time()
    # Fix vs prompt spec: use max_iter (sklearn 1.5+), not deprecated n_iter
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    result = tsne.fit_transform(embeddings)
    np.save(TSNE_SAVE, result)
    print(f"  t-SNE done in {time.time() - t0:.1f}s — saved to {TSNE_SAVE}")
    return result


def compute_umap(embeddings: np.ndarray) -> np.ndarray:
    if UMAP_SAVE.exists():
        print("  [cache] Loading UMAP from disk.")
        return np.load(UMAP_SAVE)

    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=RANDOM_SEED,
        verbose=True,
    )
    result = reducer.fit_transform(embeddings)
    np.save(UMAP_SAVE, result)
    print(f"  UMAP done in {time.time() - t0:.1f}s — saved to {UMAP_SAVE}")
    return result


# =============================================================================
# Section 4 — Plot 1: Behavior Category
# =============================================================================
def plot_behavior(tsne_2d, umap_2d, df, plots_dir) -> None:
    behavior_colors = {
        "HIGH_CONF_CONSISTENT":   "#2ecc71",
        "HIGH_CONF_INCONSISTENT": "#f39c12",
        "LOW_CONF_SCATTERED":     "#e74c3c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("V-JEPA 2 Backbone Embeddings — Prediction Behavior", fontsize=16, y=1.01)

    for ax, coords, title in zip(
        axes,
        [tsne_2d, umap_2d],
        ["t-SNE — Colored by Prediction Behavior", "UMAP — Colored by Prediction Behavior"],
    ):
        for behavior, color in behavior_colors.items():
            mask = df["behavior"] == behavior
            count = mask.sum()
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, s=8, alpha=0.7,
                label=f"{behavior} (n={count})",
            )
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=9, markerscale=2)

    plt.tight_layout()
    out = plots_dir / "behavior_clusters.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# Section 5 — Plot 2: UCF Class
# =============================================================================
def plot_ucf_classes(tsne_2d, umap_2d, df, plots_dir) -> None:
    # Fix vs prompt spec: matplotlib.colormaps instead of deprecated get_cmap
    cmap = matplotlib.colormaps["tab20"]
    classes = df["ucf_class"].unique()
    color_map = {cls: cmap(i % 20) for i, cls in enumerate(sorted(classes))}

    # Top 20 most populated classes for centroid annotation
    top20_classes = df["ucf_class"].value_counts().head(20).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle("V-JEPA 2 Backbone Embeddings — UCF-101 Classes", fontsize=16, y=1.01)

    for ax, coords, title in zip(
        axes,
        [tsne_2d, umap_2d],
        ["t-SNE — UCF-101 Classes", "UMAP — UCF-101 Classes"],
    ):
        for cls in sorted(classes):
            mask = df["ucf_class"] == cls
            c = color_map[cls]
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[c], s=6, alpha=0.6)

        # Annotate centroids for top 20 most populated classes
        for cls in top20_classes:
            mask = df["ucf_class"] == cls
            cx = coords[mask, 0].mean()
            cy = coords[mask, 1].mean()
            ax.annotate(
                cls, (cx, cy),
                fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none"),
            )

        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out = plots_dir / "ucf_class_clusters.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# Section 6 — Plot 3: Confidence Gradient
# =============================================================================
def plot_confidence(tsne_2d, umap_2d, df, plots_dir) -> None:
    confidence = df["top1_prob"].values

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("V-JEPA 2 Backbone Embeddings — Prediction Confidence", fontsize=16, y=1.01)

    for ax, coords, title in zip(
        axes,
        [tsne_2d, umap_2d],
        ["t-SNE — Prediction Confidence", "UMAP — Prediction Confidence"],
    ):
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=confidence, cmap="RdYlGn",
            s=8, alpha=0.7, vmin=0.0, vmax=1.0,
        )
        plt.colorbar(sc, ax=ax, label="Top-1 Prediction Confidence", fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out = plots_dir / "confidence_gradient.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# Section 7 — Plot 4: SSv2 Predicted Label
# =============================================================================
def plot_ssv2_labels(tsne_2d, umap_2d, df, plots_dir) -> None:
    top15_labels = df["top1_label"].value_counts().head(15).index.tolist()
    # Fix vs prompt spec: matplotlib.colormaps instead of deprecated get_cmap
    cmap = matplotlib.colormaps["tab20"]
    label_colors = {lbl: cmap(i) for i, lbl in enumerate(top15_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(
        "V-JEPA 2 Backbone Embeddings — SSv2 Predicted Labels (Top 15)",
        fontsize=16, y=1.01,
    )

    for ax, coords, title in zip(
        axes,
        [tsne_2d, umap_2d],
        ["t-SNE — SSv2 Predicted Labels (Top 15)", "UMAP — SSv2 Predicted Labels (Top 15)"],
    ):
        # Background: all other points in light gray
        bg_mask = ~df["top1_label"].isin(top15_labels)
        ax.scatter(
            coords[bg_mask, 0], coords[bg_mask, 1],
            c="lightgray", s=4, alpha=0.2, zorder=1,
        )

        # Foreground: top 15 SSv2 labels, colored
        for lbl in top15_labels:
            mask = df["top1_label"] == lbl
            short = lbl if len(lbl) <= 45 else lbl[:42] + "…"
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[label_colors[lbl]], s=10, alpha=0.8,
                label=short, zorder=2,
            )

        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=7, markerscale=1.5,
                  title="SSv2 label", title_fontsize=8)

    plt.tight_layout()
    out = plots_dir / "ssv2_label_regions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =============================================================================
# Section 8 — Summary Stats
# =============================================================================
def print_summary(tsne_2d, umap_2d, df) -> None:
    summary: dict = {}

    print("\n=== Step 2 Summary ===")

    # Per-behavior stats
    behavior_summary = {}
    for behavior in ["HIGH_CONF_CONSISTENT", "HIGH_CONF_INCONSISTENT", "LOW_CONF_SCATTERED"]:
        sub = df[df["behavior"] == behavior]
        top3_classes = sub["ucf_class"].value_counts().head(3).index.tolist()
        stats = {
            "count":           int(len(sub)),
            "mean_confidence": round(float(sub["top1_prob"].mean()), 4),
            "mean_entropy":    round(float(sub["top1_entropy"].mean()), 4),
            "top3_ucf_classes": top3_classes,
        }
        behavior_summary[behavior] = stats
        print(f"\n  {behavior}:")
        print(f"    count={stats['count']}  "
              f"mean_conf={stats['mean_confidence']:.3f}  "
              f"mean_entropy={stats['mean_entropy']:.3f}")
        print(f"    top UCF classes: {', '.join(top3_classes)}")
    summary["behavior"] = behavior_summary

    # Top 5 SSv2 labels by prediction frequency
    top5_ssv2 = df["top1_label"].value_counts().head(5)
    print(f"\n  Top 5 SSv2 labels by prediction frequency:")
    for lbl, n in top5_ssv2.items():
        print(f"    [{n:>4}]  {lbl}")
    summary["top5_ssv2_labels"] = top5_ssv2.to_dict()

    # Top 5 UCF classes by mean confidence
    class_conf = df.groupby("ucf_class")["top1_prob"].mean().sort_values(ascending=False)
    top5_conf = class_conf.head(5)
    print(f"\n  Top 5 UCF classes by mean confidence:")
    for cls, c in top5_conf.items():
        print(f"    {cls:30s}  {c:.3f}")
    summary["top5_ucf_by_confidence"] = top5_conf.to_dict()

    # Bottom 5 UCF classes by mean confidence
    bot5_conf = class_conf.tail(5)
    print(f"\n  Bottom 5 UCF classes by mean confidence:")
    for cls, c in bot5_conf.items():
        print(f"    {cls:30s}  {c:.3f}")
    summary["bottom5_ucf_by_confidence"] = bot5_conf.to_dict()

    # Save to disk
    out = EMBEDDINGS_DIR / "step2_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {out}")


# =============================================================================
# Section 9 — Entry Point
# =============================================================================
if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    assert EMBEDDINGS_NPY.exists(), "Run Step 1 first: embeddings.npy not found"
    assert METADATA_JSON.exists(), "Run Step 1 first: metadata.json not found"

    embeddings, df = load_data()

    print("\nRunning t-SNE (this may take 2-3 minutes)...")
    tsne_2d = compute_tsne(embeddings)

    print("\nRunning UMAP (this may take 1-2 minutes)...")
    umap_2d = compute_umap(embeddings)

    print("\nGenerating plots...")
    plot_behavior(tsne_2d, umap_2d, df, PLOTS_DIR)
    plot_ucf_classes(tsne_2d, umap_2d, df, PLOTS_DIR)
    plot_confidence(tsne_2d, umap_2d, df, PLOTS_DIR)
    plot_ssv2_labels(tsne_2d, umap_2d, df, PLOTS_DIR)

    print_summary(tsne_2d, umap_2d, df)

    print(f"\nStep 2 complete. Plots saved to {PLOTS_DIR}")
    print("Open the following files to review:")
    print("  results/plots/behavior_clusters.png")
    print("  results/plots/ucf_class_clusters.png")
    print("  results/plots/confidence_gradient.png")
    print("  results/plots/ssv2_label_regions.png")
    print("\nReady for Phase 3 Step 3: Nearest Neighbor Analysis")

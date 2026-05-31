# =============================================================================
# nearest_neighbor_analysis.py — Phase 3 Step 3: Nearest Neighbor Analysis
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Section 1 — Imports and Config
# =============================================================================
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Top-level constants
EMBEDDINGS_DIR     = Path("results/embeddings")
PLOTS_DIR          = Path("results/plots")
RESULTS_DIR        = Path("results")
BLIND_SPOT_CLASSES = ["PoleVault", "WritingOnBoard", "SalsaSpin", "Surfing"]
# Fix: exact label string from the data, not informal description
HITTING_LABEL      = "Hitting [something] with [something]"
RANDOM_SEED        = 42


# =============================================================================
# Section 2 — Data Loading
# =============================================================================
def load_data() -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    embeddings = np.load(EMBEDDINGS_DIR / "embeddings.npy")
    with open(EMBEDDINGS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    df       = pd.DataFrame(metadata)
    tsne_2d  = np.load(EMBEDDINGS_DIR / "tsne_2d.npy")
    umap_2d  = np.load(EMBEDDINGS_DIR / "umap_2d.npy")

    assert len(embeddings) == len(df) == len(tsne_2d) == len(umap_2d), \
        "Shape mismatch across loaded files"

    print(f"Embeddings  : {embeddings.shape}")
    print(f"t-SNE       : {tsne_2d.shape}")
    print(f"UMAP        : {umap_2d.shape}")
    print(f"UCF classes : {df['ucf_class'].nunique()}")
    print(f"SSv2 labels : {df['top1_label'].nunique()}")

    return embeddings, df, tsne_2d, umap_2d


# =============================================================================
# Section 3 — Centroid Computation
# =============================================================================
def compute_ssv2_centroids(embeddings: np.ndarray,
                            df: pd.DataFrame) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    counts = df["top1_label"].value_counts()
    eligible = counts[counts >= 5].index.tolist()
    for label in eligible:
        mask = (df["top1_label"] == label).values
        centroids[label] = embeddings[mask].mean(axis=0)

    print(f"\nSSv2 centroids computed : {len(centroids)}  (min 5 videos each)")
    print("Top 10 SSv2 label groups by size:")
    for lbl in counts.head(10).index:
        n = counts[lbl]
        marker = "*" if lbl in centroids else " "
        print(f"  {marker} [{n:>4}]  {lbl}")

    return centroids


def compute_ucf_centroids(embeddings: np.ndarray,
                           df: pd.DataFrame) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for cls in df["ucf_class"].unique():
        mask = (df["ucf_class"] == cls).values
        centroids[cls] = embeddings[mask].mean(axis=0)
    print(f"\nUCF centroids computed  : {len(centroids)}")
    return centroids


# =============================================================================
# Section 4 — Question 1: Blind Spot Analysis
# =============================================================================
def analyze_blind_spots(embeddings: np.ndarray, df: pd.DataFrame,
                         tsne_2d: np.ndarray, umap_2d: np.ndarray,
                         ssv2_centroids: dict) -> dict:
    # Fix: metric='cosine' so distances are cosine distances (1 - cosine_sim)
    nn = NearestNeighbors(n_neighbors=11, metric="cosine")
    nn.fit(embeddings)

    ssv2_labels = list(ssv2_centroids.keys())
    ssv2_matrix = np.stack([ssv2_centroids[lbl] for lbl in ssv2_labels], axis=0)

    # Fix: SSv2 centroid 2D positions = mean UMAP coords of videos in that group
    ssv2_umap_pos: dict[str, np.ndarray] = {}
    for lbl in ssv2_labels:
        mask = (df["top1_label"] == lbl).values
        ssv2_umap_pos[lbl] = umap_2d[mask].mean(axis=0)

    behavior_colors = {
        "HIGH_CONF_CONSISTENT":   "#2ecc71",
        "HIGH_CONF_INCONSISTENT": "#f39c12",
        "LOW_CONF_SCATTERED":     "#e74c3c",
    }

    findings: dict = {}
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Blind Spot Neighborhood Analysis (UMAP Space)", fontsize=16)

    for ax, cls in zip(axes.flat, BLIND_SPOT_CLASSES):
        cls_mask = (df["ucf_class"] == cls).values
        cls_embeddings = embeddings[cls_mask]
        cls_centroid   = cls_embeddings.mean(axis=0).reshape(1, -1)

        # Nearest neighbors for every video in this blind spot class
        # distances are cosine distances; col 0 is self — skip it
        distances, neighbor_indices = nn.kneighbors(cls_embeddings)
        neighbor_idx_flat  = neighbor_indices[:, 1:].flatten()
        neighbor_dist_flat = distances[:, 1:].flatten()

        # Top 20 unique nearest neighbors by smallest cosine distance
        best: dict[int, float] = {}
        for idx, dist in zip(neighbor_idx_flat, neighbor_dist_flat):
            if idx not in best or dist < best[idx]:
                best[idx] = dist
        top20 = sorted(best, key=best.get)[:20]

        neighbor_ucf  = df.iloc[top20]["ucf_class"].tolist()
        neighbor_ssv2 = df.iloc[top20]["top1_label"].tolist()
        neighbor_sims = [1.0 - best[i] for i in top20]

        # Nearest SSv2 centroids ranked by cosine similarity (higher = closer)
        ssv2_sims       = cosine_similarity(cls_centroid, ssv2_matrix)[0]
        top5_idx        = np.argsort(ssv2_sims)[::-1][:5]
        top3_idx        = top5_idx[:3]
        top5_ssv2       = [(ssv2_labels[i], float(ssv2_sims[i])) for i in top5_idx]

        n_blind_spot_nb = sum(1 for uc in neighbor_ucf if uc in BLIND_SPOT_CLASSES)
        is_isolated     = n_blind_spot_nb >= 5

        findings[cls] = {
            "nearest_ssv2_concepts":     top5_ssv2,
            "is_isolated":               is_isolated,
            "nearest_neighbor_classes":  neighbor_ucf,
            "n_blind_spot_neighbors":    n_blind_spot_nb,
        }

        print(f"\n  {cls}:")
        print(f"    nearest SSv2 concepts : {[s for s, _ in top5_ssv2[:3]]}")
        print(f"    neighbor UCF classes  : {list(dict.fromkeys(neighbor_ucf))[:5]}")
        print(f"    blind-spot neighbors  : {n_blind_spot_nb}/20 "
              f"({'isolated' if is_isolated else 'scattered'})")

        # --- Plot ---
        ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c="lightgray", s=3, alpha=0.25, zorder=1)
        nb_coords = umap_2d[top20]
        ax.scatter(nb_coords[:, 0], nb_coords[:, 1],
                   c="#3498db", s=30, alpha=0.8, zorder=3, label="Top-20 neighbors")
        ax.scatter(umap_2d[cls_mask, 0], umap_2d[cls_mask, 1],
                   c="#e74c3c", s=50, alpha=0.9, zorder=4, label=cls)

        centroid_2d = umap_2d[cls_mask].mean(axis=0)
        for i in top3_idx:
            lbl = ssv2_labels[i]
            pos = ssv2_umap_pos[lbl]
            ax.scatter(*pos, marker="*", s=350, c="#f39c12",
                       zorder=5, edgecolors="black", linewidths=0.5)
            ax.plot([centroid_2d[0], pos[0]], [centroid_2d[1], pos[1]],
                    c="#f39c12", lw=1.0, alpha=0.7, ls="--", zorder=4)
            short = lbl if len(lbl) <= 32 else lbl[:30] + "…"
            ax.annotate(short, pos, fontsize=6, ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        mean_conf = float(df[cls_mask]["top1_prob"].mean())
        ax.set_title(f"{cls}  (mean conf={mean_conf:.3f})", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, markerscale=1.5)

    plt.tight_layout()
    out = PLOTS_DIR / "blind_spot_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {out}")

    return findings


# =============================================================================
# Section 5 — Question 2: "Hitting" Absorption Analysis
# =============================================================================
def analyze_hitting_cluster(embeddings: np.ndarray, df: pd.DataFrame,
                              tsne_2d: np.ndarray, umap_2d: np.ndarray) -> dict:
    hitting_mask = (df["top1_label"] == HITTING_LABEL).values
    hitting_emb  = embeddings[hitting_mask]
    hitting_df   = df[hitting_mask].copy().reset_index(drop=True)

    print(f"\n  Videos predicted as hitting : {hitting_mask.sum()}")
    ucf_counts = hitting_df["ucf_class"].value_counts()
    top8 = ucf_counts.head(8).index.tolist()
    print(f"  UCF classes represented     : {len(ucf_counts)}")
    print(f"  Top 8                       : {top8}")

    hitting_centroid = hitting_emb.mean(axis=0).reshape(1, -1)

    # Full pairwise cosine similarity within the hitting group
    pairwise = cosine_similarity(hitting_emb)
    n = len(hitting_emb)
    upper = np.triu_indices(n, k=1)
    pair_sims = pairwise[upper]
    mean_sim  = float(pair_sims.mean())
    std_sim   = float(pair_sims.std())
    min_sim   = float(pair_sims.min())
    max_sim   = float(pair_sims.max())

    # Fix: per-class mean embeddings for the 8×8 heatmap (not raw videos)
    class_means: dict[str, np.ndarray] = {}
    for cls in top8:
        cls_mask = (hitting_df["ucf_class"] == cls).values
        class_means[cls] = hitting_emb[cls_mask].mean(axis=0)

    class_sims = {
        cls: float(cosine_similarity(class_means[cls].reshape(1, -1), hitting_centroid)[0, 0])
        for cls in top8
    }
    sorted_classes = sorted(class_sims, key=class_sims.get, reverse=True)
    most_similar   = sorted_classes[:3]
    least_similar  = sorted_classes[-3:]

    print(f"\n  Within-group mean cosine sim : {mean_sim:.4f}  (global baseline 0.93)")
    print(f"  std={std_sim:.4f}  min={min_sim:.4f}  max={max_sim:.4f}")
    print(f"  Most similar to hitting centroid  : {most_similar}")
    print(f"  Least similar to hitting centroid : {least_similar}")

    # 8×8 heatmap of per-class mean embedding similarities
    class_mean_matrix = np.stack([class_means[cls] for cls in sorted_classes], axis=0)
    heatmap_data = cosine_similarity(class_mean_matrix)
    heatmap_df   = pd.DataFrame(heatmap_data, index=sorted_classes, columns=sorted_classes)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'V-JEPA 2: "{HITTING_LABEL}" Cluster', fontsize=13)

    # Left: UMAP with hitting videos colored by UCF class
    ax = axes[0]
    ax.scatter(umap_2d[~hitting_mask, 0], umap_2d[~hitting_mask, 1],
               c="lightgray", s=4, alpha=0.3, zorder=1)
    cmap = matplotlib.colormaps["tab20"]
    for i, cls in enumerate(sorted_classes):
        cls_hit = hitting_mask & (df["ucf_class"] == cls).values
        ax.scatter(umap_2d[cls_hit, 0], umap_2d[cls_hit, 1],
                   c=[cmap(i % 20)], s=35, alpha=0.9, label=cls, zorder=2)
    other = hitting_mask & ~np.isin(df["ucf_class"].values, top8)
    if other.any():
        ax.scatter(umap_2d[other, 0], umap_2d[other, 1],
                   c="purple", s=15, alpha=0.5, label="other", zorder=2)
    ax.set_title("UMAP — Hitting Cluster by UCF Class", fontsize=12)
    ax.legend(loc="best", fontsize=7, markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Right: 8×8 per-class mean similarity heatmap
    ax = axes[1]
    sns.heatmap(
        heatmap_df, ax=ax, cmap="RdYlGn", vmin=0.5, vmax=1.0,
        annot=True, fmt=".3f", annot_kws={"size": 9},
        linewidths=0.5, cbar_kws={"label": "Cosine Similarity"},
    )
    ax.set_title("Per-class Mean Embedding Similarity\n(within Hitting group, 8×8)", fontsize=12)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "hitting_cluster.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {out}")

    return {
        "within_group_similarity":    mean_sim,
        "vs_global_baseline":         0.93,
        "mean_sim_std":               std_sim,
        "most_similar_ucf_classes":   most_similar,
        "least_similar_ucf_classes":  least_similar,
        "class_sims_to_centroid":     class_sims,
    }


# =============================================================================
# Section 6 — Question 3: Consistency vs Scatter Geometry
# =============================================================================
def analyze_behavior_geometry(embeddings: np.ndarray, df: pd.DataFrame,
                               ucf_centroids: dict,
                               ssv2_centroids: dict) -> dict:
    ssv2_labels = list(ssv2_centroids.keys())
    ssv2_matrix = np.stack([ssv2_centroids[lbl] for lbl in ssv2_labels], axis=0)

    behavior_colors = {
        "HIGH_CONF_CONSISTENT":   "#2ecc71",
        "HIGH_CONF_INCONSISTENT": "#f39c12",
        "LOW_CONF_SCATTERED":     "#e74c3c",
    }

    rows = []
    for cls, centroid in ucf_centroids.items():
        c2d      = centroid.reshape(1, -1)
        cls_mask = (df["ucf_class"] == cls).values
        cls_emb  = embeddings[cls_mask]
        cls_df   = df[cls_mask]

        ssv2_sims  = cosine_similarity(c2d, ssv2_matrix)[0]
        sorted_idx = np.argsort(ssv2_sims)[::-1]

        sim_nearest = float(ssv2_sims[sorted_idx[0]])
        sim_2nd     = float(ssv2_sims[sorted_idx[1]])

        # Fix: ambiguity uses cosine DISTANCE (1 - sim), not raw similarity
        dist_nearest = 1.0 - sim_nearest
        dist_2nd     = 1.0 - sim_2nd
        ambiguity    = dist_nearest / dist_2nd if dist_2nd > 1e-8 else 0.0

        # Intra-class spread = mean pairwise cosine distance within class
        if len(cls_emb) >= 2:
            pw      = cosine_similarity(cls_emb)
            n       = len(cls_emb)
            upper   = np.triu_indices(n, k=1)
            intra_spread = float((1.0 - pw[upper]).mean())
        else:
            intra_spread = 0.0

        rows.append({
            "ucf_class":      cls,
            "behavior":       cls_df["behavior"].iloc[0],
            "ambiguity":      ambiguity,
            "intra_spread":   intra_spread,
            "mean_top1_prob": float(cls_df["top1_prob"].mean()),
            "nearest_ssv2":   ssv2_labels[sorted_idx[0]],
            "second_ssv2":    ssv2_labels[sorted_idx[1]],
            "sim_nearest":    sim_nearest,
            "sim_2nd":        sim_2nd,
        })

    geo_df = pd.DataFrame(rows)

    corr_ambiguity = float(np.corrcoef(geo_df["ambiguity"],    geo_df["mean_top1_prob"])[0, 1])
    corr_spread    = float(np.corrcoef(geo_df["intra_spread"], geo_df["mean_top1_prob"])[0, 1])

    most_ambiguous  = geo_df.nlargest(5, "ambiguity")["ucf_class"].tolist()
    least_ambiguous = geo_df.nsmallest(5, "ambiguity")["ucf_class"].tolist()

    print(f"\n  Pearson r (ambiguity vs confidence) : {corr_ambiguity:.4f}")
    print(f"  Pearson r (spread vs confidence)    : {corr_spread:.4f}")
    print(f"  Most ambiguous  UCF classes : {most_ambiguous}")
    print(f"  Least ambiguous UCF classes : {least_ambiguous}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Behavior Geometry: Ambiguity and Spread vs Confidence", fontsize=14)

    for ax, x_col, corr_val, x_label, title in zip(
        axes,
        ["ambiguity",    "intra_spread"],
        [corr_ambiguity, corr_spread],
        ["Ambiguity Score  (cosine dist to nearest / 2nd nearest SSv2 centroid)",
         "Intra-class Spread  (mean pairwise cosine distance)"],
        ["Ambiguity Score vs Mean Confidence",
         "Intra-class Spread vs Mean Confidence"],
    ):
        for behavior, color in behavior_colors.items():
            sub = geo_df[geo_df["behavior"] == behavior]
            ax.scatter(sub[x_col], sub["mean_top1_prob"],
                       c=color, s=60, alpha=0.7, label=behavior, zorder=2)

        if x_col == "ambiguity":
            for _, row in geo_df.nlargest(5, "ambiguity").iterrows():
                ax.annotate(row["ucf_class"],
                            (row["ambiguity"], row["mean_top1_prob"]),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points")
            for _, row in geo_df.nsmallest(5, "ambiguity").iterrows():
                ax.annotate(row["ucf_class"],
                            (row["ambiguity"], row["mean_top1_prob"]),
                            fontsize=7, ha="right", va="top",
                            xytext=(-3, -3), textcoords="offset points")

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("Mean Top-1 Confidence", fontsize=11)
        ax.set_title(f"{title}  (r={corr_val:.3f})", fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "behavior_geometry.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {out}")

    return {
        "ambiguity_correlation_with_confidence": corr_ambiguity,
        "spread_correlation_with_confidence":    corr_spread,
        "most_ambiguous_classes":                most_ambiguous,
        "least_ambiguous_classes":               least_ambiguous,
        "per_class":                             geo_df.to_dict(orient="records"),
    }


# =============================================================================
# Section 7 — Findings Report
# =============================================================================
def save_findings(blind_spot_findings: dict,
                  hitting_findings: dict,
                  behavior_findings: dict) -> None:

    def _clean(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    blind_json = {
        cls: _clean({
            "nearest_ssv2_concepts":    [(lbl, sim) for lbl, sim in data["nearest_ssv2_concepts"]],
            "is_isolated":               data["is_isolated"],
            "nearest_neighbor_classes":  data["nearest_neighbor_classes"],
            "n_blind_spot_neighbors":    data["n_blind_spot_neighbors"],
        })
        for cls, data in blind_spot_findings.items()
    }

    hitting_json = _clean({
        k: v for k, v in hitting_findings.items() if k != "class_sims_to_centroid"
    })

    behavior_json = _clean({
        k: v for k, v in behavior_findings.items() if k != "per_class"
    })

    findings = {
        "blind_spots":       blind_json,
        "hitting_cluster":   hitting_json,
        "behavior_geometry": behavior_json,
    }

    out = RESULTS_DIR / "step3_findings.json"
    with open(out, "w") as f:
        json.dump(findings, f, indent=2)

    print(f"\n=== Step 3 Findings Summary ===")
    print(f"\nBlind spots:")
    for cls, data in findings["blind_spots"].items():
        top1 = data["nearest_ssv2_concepts"][0][0]
        print(f"  {cls:20s}  isolated={data['is_isolated']}  "
              f"nearest SSv2: {top1[:50]}")
    hc = findings["hitting_cluster"]
    print(f"\nHitting cluster:")
    print(f"  within-group sim={hc['within_group_similarity']:.4f}  (baseline 0.93)")
    print(f"  most similar  : {hc['most_similar_ucf_classes']}")
    print(f"  least similar : {hc['least_similar_ucf_classes']}")
    bg = findings["behavior_geometry"]
    print(f"\nBehavior geometry:")
    print(f"  ambiguity r={bg['ambiguity_correlation_with_confidence']:.4f}  "
          f"spread r={bg['spread_correlation_with_confidence']:.4f}")
    print(f"  most ambiguous  : {bg['most_ambiguous_classes']}")
    print(f"  least ambiguous : {bg['least_ambiguous_classes']}")
    print(f"\n  Findings saved to {out}")


# =============================================================================
# Section 8 — Entry Point
# =============================================================================
if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings, df, tsne_2d, umap_2d = load_data()

    ssv2_centroids = compute_ssv2_centroids(embeddings, df)
    ucf_centroids  = compute_ucf_centroids(embeddings, df)

    print("\n--- Question 1: Blind Spot Analysis ---")
    blind_spot_findings = analyze_blind_spots(
        embeddings, df, tsne_2d, umap_2d, ssv2_centroids)

    print("\n--- Question 2: Hitting Absorption Analysis ---")
    hitting_findings = analyze_hitting_cluster(
        embeddings, df, tsne_2d, umap_2d)

    print("\n--- Question 3: Behavior Geometry Analysis ---")
    behavior_findings = analyze_behavior_geometry(
        embeddings, df, ucf_centroids, ssv2_centroids)

    save_findings(blind_spot_findings, hitting_findings, behavior_findings)

    print(f"\nStep 3 complete. Results saved to results/step3_findings.json")
    print("New plots saved to:")
    print("  results/plots/blind_spot_analysis.png")
    print("  results/plots/hitting_cluster.png")
    print("  results/plots/behavior_geometry.png")
    print("\nReady for Phase 3 Step 4: Behavior Geometry Analysis")

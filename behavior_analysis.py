"""Phase 3 Step 4 — Behavior Analysis: Testing three hypotheses about what drives
V-JEPA 2 prediction confidence and behavior category.

Pure numpy/pandas/scipy/sklearn — no torch, no model loading.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway, pointbiserialr
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDINGS_DIR = Path("results/embeddings")
PLOTS_DIR = Path("results/plots")
RESULTS_DIR = Path("results")

BEHAVIOR_COLORS = {
    "HIGH_CONF_CONSISTENT": "#2ecc71",
    "HIGH_CONF_INCONSISTENT": "#f39c12",
    "LOW_CONF_SCATTERED": "#e74c3c",
}

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Section 2 — Data Loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(EMBEDDINGS_DIR / "embeddings.npy")
    metadata = json.loads((EMBEDDINGS_DIR / "metadata.json").read_text())
    df = pd.DataFrame(metadata)

    assert len(embeddings) == len(df), (
        f"Shape mismatch: {len(embeddings)} embeddings vs {len(df)} metadata rows"
    )

    behavior_counts = df.groupby("behavior")["ucf_class"].nunique()
    print(f"Embeddings  : {embeddings.shape}")
    print(f"UCF classes : {df['ucf_class'].nunique()}")
    print(f"Behavior category counts:")
    for cat, cnt in behavior_counts.items():
        print(f"  {cat}: {cnt} classes")

    return embeddings, df


# ---------------------------------------------------------------------------
# Section 3 — Per-Class Feature Computation
# ---------------------------------------------------------------------------

def compute_class_features(embeddings: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    # Global SSv2 center approximation — mean of all 1059 embeddings
    global_center = embeddings.mean(axis=0)

    rows = []
    for ucf_class, group in df.groupby("ucf_class"):
        idx = group["embedding_idx"].values
        X = embeddings[idx].astype(np.float64)
        n = len(X)

        # Feature 1 — intra_class_variance (proxy: temporal dynamics)
        intra_class_variance = float(np.var(X, axis=0).mean())

        # Feature 2 — distance_from_ssv2_center (proxy: spatial scale)
        centroid = X.mean(axis=0)
        sim_to_center = cosine_similarity(
            centroid.reshape(1, -1), global_center.reshape(1, -1)
        )[0, 0]
        distance_from_ssv2_center = float(1.0 - sim_to_center)

        # Feature 3 — within_class_similarity (proxy: cluster tightness)
        # Bug fix: use upper triangle only (k=1) to exclude self-similarity diagonal
        if n < 2:
            within_class_similarity = float("nan")
        else:
            sim_mat = cosine_similarity(X)
            triu_idx = np.triu_indices(n, k=1)
            within_class_similarity = float(sim_mat[triu_idx].mean())

        # Feature 4 — embedding_norm
        embedding_norm = float(np.linalg.norm(X, axis=1).mean())

        # Feature 5 — confidence statistics
        top1_prob_mean = float(group["top1_prob"].mean())
        top1_prob_std = float(group["top1_prob"].std())

        behavior = group["behavior"].iloc[0]

        rows.append({
            "ucf_class": ucf_class,
            "behavior": behavior,
            "intra_class_variance": intra_class_variance,
            "distance_from_ssv2_center": distance_from_ssv2_center,
            "within_class_similarity": within_class_similarity,
            "embedding_norm": embedding_norm,
            "top1_prob_mean": top1_prob_mean,
            "top1_prob_std": top1_prob_std,
            "n_videos": n,
        })

    class_df = pd.DataFrame(rows).sort_values("ucf_class").reset_index(drop=True)
    class_df.to_csv(RESULTS_DIR / "class_features.csv", index=False)
    print(f"\nClass features saved to results/class_features.csv  ({len(class_df)} rows)")
    return class_df


# ---------------------------------------------------------------------------
# Section 4 — Hypothesis Testing
# ---------------------------------------------------------------------------

def _verdict(pearson_r: float, pearson_p: float, anova_p: float) -> str:
    if abs(pearson_r) > 0.3 and pearson_p < 0.05 and anova_p < 0.05:
        return "SUPPORTED"
    if abs(pearson_r) > 0.15 and pearson_p < 0.1:
        return "PARTIALLY SUPPORTED"
    return "REJECTED"


def test_hypotheses(class_df: pd.DataFrame) -> dict:
    binary_consistent = (class_df["behavior"] == "HIGH_CONF_CONSISTENT").astype(int).values
    confidence = class_df["top1_prob_mean"].values

    hypotheses = [
        ("A", "Temporal Dynamics Proxy", "intra_class_variance"),
        ("B", "Spatial Scale Proxy",     "distance_from_ssv2_center"),
        ("C", "Cluster Tightness",       "within_class_similarity"),
    ]

    results = {}
    for key, label, feature_col in hypotheses:
        feat = class_df[feature_col].dropna().values
        # Use rows without NaN for all stats
        valid_mask = ~class_df[feature_col].isna()
        feat_v = class_df.loc[valid_mask, feature_col].values
        conf_v = class_df.loc[valid_mask, "top1_prob_mean"].values
        binary_v = binary_consistent[valid_mask.values]
        behavior_v = class_df.loc[valid_mask, "behavior"].values

        pearson_r, pearson_p = pearsonr(feat_v, conf_v)
        pb_r, pb_p = pointbiserialr(binary_v, feat_v)

        groups = [
            feat_v[behavior_v == cat]
            for cat in ["HIGH_CONF_CONSISTENT", "HIGH_CONF_INCONSISTENT", "LOW_CONF_SCATTERED"]
        ]
        anova_f, anova_p = f_oneway(*groups)

        verdict = _verdict(pearson_r, pearson_p, anova_p)

        print(f"\nHYPOTHESIS {key} — {label} ({feature_col})")
        print(f"  Pearson r with confidence:        r = {pearson_r:+.4f}, p = {pearson_p:.4f}")
        print(f"  Point-biserial (consistent=1):    r = {pb_r:+.4f}, p = {pb_p:.4f}")
        print(f"  ANOVA across behavior categories: F = {anova_f:.4f}, p = {anova_p:.4f}")
        print(f"  Verdict: {verdict}")

        results[f"hypothesis_{key.lower()}"] = {
            "proxy": feature_col,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "pointbiserial_r": float(pb_r),
            "pointbiserial_p": float(pb_p),
            "anova_f": float(anova_f),
            "anova_p": float(anova_p),
            "verdict": verdict,
        }

    return results


# ---------------------------------------------------------------------------
# Section 5 — Visualizations
# ---------------------------------------------------------------------------

def _annotate_residuals(ax, x_vals, y_vals, labels, n=5):
    """Annotate the n points with largest absolute residual from regression line."""
    r, _ = pearsonr(x_vals, y_vals)
    # Standardize to compute residuals on same scale
    x_z = (x_vals - x_vals.mean()) / (x_vals.std() + 1e-12)
    y_z = (y_vals - y_vals.mean()) / (y_vals.std() + 1e-12)
    # OLS slope = r (when both are z-scored)
    residuals = np.abs(y_z - r * x_z)
    top_idx = np.argsort(residuals)[::-1][:n]
    for i in top_idx:
        ax.annotate(
            labels[i],
            (x_vals[i], y_vals[i]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )


def _scatter_with_regression(ax, class_df, feature_col, hyp_results_entry):
    valid = class_df[feature_col].notna()
    sub = class_df[valid]
    x = sub[feature_col].values
    y = sub["top1_prob_mean"].values
    labels = sub["ucf_class"].values
    behaviors = sub["behavior"].values

    for beh, color in BEHAVIOR_COLORS.items():
        mask = behaviors == beh
        ax.scatter(x[mask], y[mask], c=color, label=beh, alpha=0.75, s=40, edgecolors="white", lw=0.3)

    # Regression line
    m = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, np.polyval(m, x_line), color="black", lw=1, ls="--", alpha=0.5)

    r = hyp_results_entry["pearson_r"]
    p = hyp_results_entry["pearson_p"]
    ax.text(
        0.04, 0.96,
        f"r = {r:+.3f}  p = {p:.3f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    _annotate_residuals(ax, x, y, labels, n=5)
    ax.set_ylabel("Mean top-1 probability", fontsize=9)
    ax.legend(fontsize=7, markerscale=0.8)


def generate_plots(class_df: pd.DataFrame, hypothesis_results: dict) -> None:
    behavior_order = list(BEHAVIOR_COLORS.keys())
    palette = BEHAVIOR_COLORS

    # ------------------------------------------------------------------
    # Plot 8 — Hypothesis A: Intra-class Variance
    # ------------------------------------------------------------------
    fig, (ax_v, ax_s) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    sns.violinplot(
        data=class_df, x="behavior", y="intra_class_variance",
        hue="behavior", order=behavior_order, palette=palette, cut=0, legend=False, ax=ax_v,
    )
    ax_v.set_title("Hypothesis A: Intra-class Variance by Behavior", fontsize=11)
    ax_v.set_xlabel("Behavior category", fontsize=9)
    ax_v.set_ylabel("Intra-class variance (mean per-dim)", fontsize=9)
    ax_v.tick_params(axis="x", labelsize=7)

    _scatter_with_regression(ax_s, class_df, "intra_class_variance", hypothesis_results["hypothesis_a"])
    ax_s.set_xlabel("Intra-class variance", fontsize=9)
    ax_s.set_title("Hypothesis A: Variance vs Confidence", fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hypothesis_a_variance.png")
    plt.close(fig)
    print("  Saved results/plots/hypothesis_a_variance.png")

    # ------------------------------------------------------------------
    # Plot 9 — Hypothesis B: Distance from SSv2 Center
    # ------------------------------------------------------------------
    fig, (ax_v, ax_s) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    sns.violinplot(
        data=class_df, x="behavior", y="distance_from_ssv2_center",
        hue="behavior", order=behavior_order, palette=palette, cut=0, legend=False, ax=ax_v,
    )
    ax_v.set_title("Hypothesis B: Distance from SSv2 Center by Behavior", fontsize=11)
    ax_v.set_xlabel("Behavior category", fontsize=9)
    ax_v.set_ylabel("Cosine distance from global center", fontsize=9)
    ax_v.tick_params(axis="x", labelsize=7)

    _scatter_with_regression(ax_s, class_df, "distance_from_ssv2_center", hypothesis_results["hypothesis_b"])
    ax_s.set_xlabel("Distance from SSv2 center", fontsize=9)
    ax_s.set_title("Hypothesis B: SSv2 Distance vs Confidence", fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hypothesis_b_distance.png")
    plt.close(fig)
    print("  Saved results/plots/hypothesis_b_distance.png")

    # ------------------------------------------------------------------
    # Plot 10 — Hypothesis C: Within-class Similarity
    # ------------------------------------------------------------------
    fig, (ax_v, ax_s) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    sns.violinplot(
        data=class_df.dropna(subset=["within_class_similarity"]),
        x="behavior", y="within_class_similarity",
        hue="behavior", order=behavior_order, palette=palette, cut=0, legend=False, ax=ax_v,
    )
    ax_v.set_title("Hypothesis C: Within-class Similarity by Behavior", fontsize=11)
    ax_v.set_xlabel("Behavior category", fontsize=9)
    ax_v.set_ylabel("Mean pairwise cosine similarity", fontsize=9)
    ax_v.tick_params(axis="x", labelsize=7)

    _scatter_with_regression(ax_s, class_df, "within_class_similarity", hypothesis_results["hypothesis_c"])
    ax_s.set_xlabel("Within-class cosine similarity", fontsize=9)
    ax_s.set_title("Hypothesis C: Within-class Similarity vs Confidence", fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hypothesis_c_similarity.png")
    plt.close(fig)
    print("  Saved results/plots/hypothesis_c_similarity.png")

    # ------------------------------------------------------------------
    # Plot 11 — Combined Feature Summary
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.suptitle(
        "What drives V-JEPA 2 prediction behavior on UCF-101?",
        fontsize=13, fontweight="bold",
    )

    feature_labels = [
        ("intra_class_variance",      "Intra-class variance\n(temporal proxy)"),
        ("distance_from_ssv2_center", "Distance from SSv2 center\n(spatial scale proxy)"),
        ("within_class_similarity",   "Within-class similarity\n(cluster tightness proxy)"),
    ]

    for ax, (feat, ylabel) in zip(axes, feature_labels):
        sub = class_df.dropna(subset=[feat])
        sns.boxplot(
            data=sub, x="behavior", y=feat,
            hue="behavior", order=behavior_order, palette=palette, legend=False, ax=ax,
        )
        ax.set_title(ylabel, fontsize=9)
        ax.set_xlabel("Behavior category", fontsize=8)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_summary.png")
    plt.close(fig)
    print("  Saved results/plots/feature_summary.png")


# ---------------------------------------------------------------------------
# Section 6 — Step 4 Findings Summary
# ---------------------------------------------------------------------------

def save_step4_findings(class_df: pd.DataFrame, hypothesis_results: dict) -> None:
    # Top predictive feature by |Pearson r|
    top_feature = max(
        ["hypothesis_a", "hypothesis_b", "hypothesis_c"],
        key=lambda k: abs(hypothesis_results[k]["pearson_r"]),
    )
    top_feature_name = hypothesis_results[top_feature]["proxy"]

    # Classes best explained: HIGH_CONF_CONSISTENT with above-median within_class_similarity
    valid = class_df.dropna(subset=["within_class_similarity"])
    median_sim = valid["within_class_similarity"].median()
    best_explained = valid[
        (valid["behavior"] == "HIGH_CONF_CONSISTENT") &
        (valid["within_class_similarity"] > median_sim)
    ]["ucf_class"].tolist()

    # Classes unexplained: LOW_CONF_SCATTERED with above-median within_class_similarity
    unexplained = valid[
        (valid["behavior"] == "LOW_CONF_SCATTERED") &
        (valid["within_class_similarity"] > median_sim)
    ]["ucf_class"].tolist()

    # Build overall_conclusion from verdicts
    verdicts = {
        "A": hypothesis_results["hypothesis_a"]["verdict"],
        "B": hypothesis_results["hypothesis_b"]["verdict"],
        "C": hypothesis_results["hypothesis_c"]["verdict"],
    }
    supported = [k for k, v in verdicts.items() if v == "SUPPORTED"]
    partial = [k for k, v in verdicts.items() if v == "PARTIALLY SUPPORTED"]
    rejected = [k for k, v in verdicts.items() if v == "REJECTED"]

    proxy_map = {
        "A": "temporal dynamics (intra-class embedding variance)",
        "B": "spatial scale (distance from SSv2 distribution center)",
        "C": "cluster tightness (within-class cosine similarity)",
    }

    if supported:
        lead = f"V-JEPA 2's prediction confidence on UCF-101 is most strongly explained by {', '.join(proxy_map[k] for k in supported)}."
    elif partial:
        lead = f"No hypothesis was strongly confirmed; {', '.join(proxy_map[k] for k in partial)} showed a weak association with confidence."
    else:
        lead = "None of the three embedding-derived proxies strongly predicted confidence or behavior category."

    if rejected:
        follow = f"The {', '.join(proxy_map[k] for k in rejected)} hypothesis was rejected, suggesting that factor alone does not drive the HIGH/LOW confidence split."
    else:
        follow = "All proxies showed at least partial associations, suggesting confidence is multi-factorial."

    coda = (
        f"The strongest single predictor was {top_feature_name}, "
        f"while {len(unexplained)} LOW_CONF_SCATTERED classes with unexpectedly tight clusters "
        "remain unexplained by embedding geometry alone."
    )
    overall_conclusion = f"{lead} {follow} {coda}"

    findings = {
        "hypothesis_a": hypothesis_results["hypothesis_a"],
        "hypothesis_b": hypothesis_results["hypothesis_b"],
        "hypothesis_c": hypothesis_results["hypothesis_c"],
        "overall_conclusion": overall_conclusion,
        "top_predictive_feature": top_feature_name,
        "classes_best_explained": best_explained,
        "classes_unexplained": unexplained,
    }

    out_path = RESULTS_DIR / "step4_findings.json"
    out_path.write_text(json.dumps(findings, indent=2))

    print(f"\n=== Overall Conclusion ===")
    print(overall_conclusion)


# ---------------------------------------------------------------------------
# Section 7 — Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings, df = load_data()
    class_df = compute_class_features(embeddings, df)

    print("\n--- Testing Three Hypotheses ---")
    hypothesis_results = test_hypotheses(class_df)

    print("\n--- Generating Plots ---")
    generate_plots(class_df, hypothesis_results)

    save_step4_findings(class_df, hypothesis_results)

    print(f"\nStep 4 complete.")
    print("Results saved to results/step4_findings.json")
    print("New plots saved to:")
    print("  results/plots/hypothesis_a_variance.png")
    print("  results/plots/hypothesis_b_distance.png")
    print("  results/plots/hypothesis_c_similarity.png")
    print("  results/plots/feature_summary.png")
    print("\nReady for Phase 3 Step 5: Final Report")

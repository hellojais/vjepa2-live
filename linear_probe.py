"""Phase 4 — Linear Probe Experiment.

Tests whether the frozen V-JEPA 2 backbone already encodes sufficient
information to classify UCF-101 actions — with only the output mapping changed.

Pure sklearn/numpy — no torch, no model loading.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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

BLIND_SPOT_CLASSES = ["PoleVault", "WritingOnBoard", "SalsaSpin", "Surfing"]
N_FOLDS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Section 1 — Data Loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    embeddings = np.load(EMBEDDINGS_DIR / "embeddings.npy")
    metadata = json.loads((EMBEDDINGS_DIR / "metadata.json").read_text())
    df = pd.DataFrame(metadata)
    class_features = pd.read_csv(RESULTS_DIR / "class_features.csv")

    assert len(embeddings) == len(df)

    print(f"Embeddings      : {embeddings.shape}")
    print(f"UCF classes     : {df['ucf_class'].nunique()}")
    print(f"Videos per class: min={df.groupby('ucf_class').size().min()}  "
          f"max={df.groupby('ucf_class').size().max()}")

    labels = df["ucf_class"].values
    return embeddings, labels, df, class_features


# ---------------------------------------------------------------------------
# Section 2 — Cross-validated Linear Probe
# ---------------------------------------------------------------------------

def run_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
) -> tuple[float, float, dict[str, float]]:
    """5-fold stratified cross-validation. Returns (mean_acc, std_acc, per_class_acc)."""

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    # Accumulate per-class correct/total across folds
    class_correct: dict[str, int] = {c: 0 for c in np.unique(labels)}
    class_total: dict[str, int] = {c: 0 for c in np.unique(labels)}

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Fit scaler on train only — never touch test data
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)

        fold_acc = (preds == y_test).mean()
        fold_accuracies.append(fold_acc)
        print(f"  Fold {fold + 1}/5 accuracy: {fold_acc:.4f}")

        for true_label, pred_label in zip(y_test, preds):
            class_total[true_label] += 1
            if true_label == pred_label:
                class_correct[true_label] += 1

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    per_class_acc = {
        cls: (class_correct[cls] / class_total[cls] if class_total[cls] > 0 else float("nan"))
        for cls in class_correct
    }

    return mean_acc, std_acc, per_class_acc


# ---------------------------------------------------------------------------
# Section 3 — Derived Statistics
# ---------------------------------------------------------------------------

def compute_statistics(
    per_class_acc: dict[str, float],
    class_features: pd.DataFrame,
) -> dict:
    cf = class_features.copy()
    cf["linear_probe_acc"] = cf["ucf_class"].map(per_class_acc)

    # Bug fix 1: use top1_prob_mean (actual CSV column name)
    cf["delta"] = cf["linear_probe_acc"] - cf["top1_prob_mean"]

    # Per-behavior accuracy
    per_behavior = (
        cf.groupby("behavior")["linear_probe_acc"].mean().to_dict()
    )

    # Blind spot accuracy
    blind_spot_acc = {
        cls: per_class_acc.get(cls, float("nan"))
        for cls in BLIND_SPOT_CLASSES
    }

    # Tight cluster paradox — load unexplained classes from step4 findings
    step4 = json.loads((RESULTS_DIR / "step4_findings.json").read_text())
    paradox_classes = step4["classes_unexplained"]
    paradox_df = cf[cf["ucf_class"].isin(paradox_classes)]
    mean_probe_acc = float(paradox_df["linear_probe_acc"].mean())
    mean_ssv2_conf = float(paradox_df["top1_prob_mean"].mean())
    paradox_delta = mean_probe_acc - mean_ssv2_conf

    if paradox_delta > 0.2:
        paradox_conclusion = (
            "Backbone encodes UCF-101 action information for the tight-cluster paradox classes "
            f"(linear probe {mean_probe_acc:.3f} vs SSv2 confidence {mean_ssv2_conf:.3f}, "
            f"delta={paradox_delta:+.3f}). The SSv2 classification head is the bottleneck, "
            "not the representation."
        )
    elif paradox_delta > 0.0:
        paradox_conclusion = (
            "Backbone shows modest improvement over SSv2 confidence for tight-cluster paradox classes "
            f"(linear probe {mean_probe_acc:.3f} vs SSv2 confidence {mean_ssv2_conf:.3f}, "
            f"delta={paradox_delta:+.3f}). Representation quality is partially sufficient."
        )
    else:
        paradox_conclusion = (
            "Even a linear probe on the backbone cannot classify the tight-cluster paradox classes "
            f"better than SSv2 confidence (linear probe {mean_probe_acc:.3f} vs SSv2 "
            f"confidence {mean_ssv2_conf:.3f}, delta={paradox_delta:+.3f}). "
            "The representation itself is deficient for these classes."
        )

    # Bug fix 3: store top10 lists as dicts with all relevant fields
    cf_sorted = cf.dropna(subset=["delta"]).sort_values("delta", ascending=False)
    top10_most_improved = [
        {
            "ucf_class": row["ucf_class"],
            "linear_probe_acc": float(row["linear_probe_acc"]),
            "ssv2_confidence": float(row["top1_prob_mean"]),
            "delta": float(row["delta"]),
        }
        for _, row in cf_sorted.head(10).iterrows()
    ]
    top10_least_improved = [
        {
            "ucf_class": row["ucf_class"],
            "linear_probe_acc": float(row["linear_probe_acc"]),
            "ssv2_confidence": float(row["top1_prob_mean"]),
            "delta": float(row["delta"]),
        }
        for _, row in cf_sorted.tail(10).sort_values("delta").iterrows()
    ]

    return {
        "per_class_acc": per_class_acc,
        "per_behavior": per_behavior,
        "blind_spot_acc": blind_spot_acc,
        "paradox": {
            "classes": paradox_classes,
            "mean_linear_probe_accuracy": mean_probe_acc,
            "mean_ssv2_confidence": mean_ssv2_conf,
            "delta": paradox_delta,
            "conclusion": paradox_conclusion,
        },
        "top10_most_improved": top10_most_improved,
        "top10_least_improved": top10_least_improved,
        "cf_with_delta": cf,
    }


# ---------------------------------------------------------------------------
# Section 4 — Plot
# ---------------------------------------------------------------------------

def generate_plot(stats: dict, mean_acc: float, std_acc: float) -> None:
    cf = stats["cf_with_delta"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=150)
    fig.suptitle(
        "Linear Probe vs Zero-Shot SSv2: Does the Backbone Have the Information?",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ---- Top left: per-behavior bar chart ----
    ax = axes[0, 0]
    behaviors = list(BEHAVIOR_COLORS.keys())
    accs = [stats["per_behavior"].get(b, 0.0) for b in behaviors]
    colors = [BEHAVIOR_COLORS[b] for b in behaviors]
    bars = ax.bar(behaviors, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean linear probe accuracy", fontsize=10)
    ax.set_title("Per-Behavior Linear Probe Accuracy", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    # ---- Top right: scatter SSv2 confidence vs linear probe accuracy ----
    ax = axes[0, 1]
    for beh, color in BEHAVIOR_COLORS.items():
        mask = cf["behavior"] == beh
        ax.scatter(
            cf.loc[mask, "top1_prob_mean"],
            cf.loc[mask, "linear_probe_acc"],
            c=color, label=beh, alpha=0.75, s=45,
            edgecolors="white", linewidths=0.3,
        )
    # Bug fix 2: diagonal is numerical equivalence, not performance comparison
    lim = (0, 1)
    ax.plot(lim, lim, color="gray", lw=1, ls="--", alpha=0.6,
            label="linear probe acc = SSv2 confidence\n(numerical equivalence, not performance comparison)")
    # Annotate blind spot classes
    for cls in BLIND_SPOT_CLASSES:
        row = cf[cf["ucf_class"] == cls]
        if len(row):
            ax.annotate(
                cls,
                (row["top1_prob_mean"].values[0], row["linear_probe_acc"].values[0]),
                fontsize=7, xytext=(5, 5), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Zero-shot SSv2 confidence (top1_prob_mean)", fontsize=10)
    ax.set_ylabel("Linear probe accuracy", fontsize=10)
    ax.set_title("SSv2 Confidence vs Linear Probe Accuracy per Class", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")

    # ---- Bottom left: top 15 most improved (green) ----
    ax = axes[1, 0]
    most = sorted(stats["top10_most_improved"], key=lambda x: x["delta"])
    # Extend to 15 if we have enough data
    cf_sorted = cf.dropna(subset=["delta"]).sort_values("delta", ascending=False)
    top15_most = cf_sorted.head(15)
    top15_most = top15_most.sort_values("delta")
    ax.barh(
        top15_most["ucf_class"],
        top15_most["delta"],
        color="#2ecc71", edgecolor="white", linewidth=0.5,
    )
    ax.axvline(0, color="black", lw=0.8, ls="-")
    ax.set_xlabel("Delta (linear probe acc − SSv2 confidence)", fontsize=9)
    ax.set_title("Top 15 Most Improved Classes\n(backbone has info, SSv2 head is bottleneck)", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)

    # ---- Bottom right: top 15 least improved (red) ----
    ax = axes[1, 1]
    bottom15_least = cf_sorted.tail(15).sort_values("delta", ascending=False)
    ax.barh(
        bottom15_least["ucf_class"],
        bottom15_least["delta"],
        color="#e74c3c", edgecolor="white", linewidth=0.5,
    )
    ax.axvline(0, color="black", lw=0.8, ls="-")
    ax.set_xlabel("Delta (linear probe acc − SSv2 confidence)", fontsize=9)
    ax.set_title("Top 15 Least Improved Classes\n(representation may be deficient)", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = PLOTS_DIR / "linear_probe.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Section 5 — Save Results & Print Summary
# ---------------------------------------------------------------------------

def save_results(
    stats: dict,
    mean_acc: float,
    std_acc: float,
) -> None:
    per_class_acc = stats["per_class_acc"]

    results = {
        "overall_accuracy_mean": mean_acc,
        "overall_accuracy_std": std_acc,
        "per_class_accuracy": {k: float(v) for k, v in per_class_acc.items()},
        "per_behavior_accuracy": stats["per_behavior"],
        "blind_spot_accuracy": stats["blind_spot_acc"],
        "tight_cluster_paradox": {
            "mean_linear_probe_accuracy": stats["paradox"]["mean_linear_probe_accuracy"],
            "mean_ssv2_confidence": stats["paradox"]["mean_ssv2_confidence"],
            "delta": stats["paradox"]["delta"],
            "conclusion": stats["paradox"]["conclusion"],
        },
        "top10_most_improved_classes": stats["top10_most_improved"],
        "top10_least_improved_classes": stats["top10_least_improved"],
        "methodology_note": (
            "~10 samples per class, 101-class problem. "
            "Per-class accuracy is coarse (2 test samples per fold for most classes). "
            "Delta compares linear probe correctness (UCF labels) against SSv2 softmax "
            "confidence (SSv2 labels) — these are incommensurable as absolute scores but "
            "valid for relative comparison: high delta = backbone encodes UCF structure "
            "that the SSv2 head cannot express. "
            "Results are internally valid for relative comparison but not representative "
            "of large-sample linear probe benchmarks."
        ),
    }

    out_path = RESULTS_DIR / "linear_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    # Print summary
    print(f"\n=== Linear Probe Summary ===")
    print(f"Overall accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    print(f"\nPer-behavior accuracy:")
    for beh, acc in stats["per_behavior"].items():
        print(f"  {beh:<28}: {acc:.4f}")

    print(f"\nBlind spot class accuracies:")
    for cls, acc in stats["blind_spot_acc"].items():
        print(f"  {cls:<20}: {acc:.4f}")

    print(f"\nTight cluster paradox:")
    print(f"  {stats['paradox']['conclusion']}")

    print(f"\nTop 5 most improved classes (backbone has info, SSv2 head is bottleneck):")
    for entry in stats["top10_most_improved"][:5]:
        print(f"  {entry['ucf_class']:<25}  delta={entry['delta']:+.4f}  "
              f"probe={entry['linear_probe_acc']:.3f}  ssv2_conf={entry['ssv2_confidence']:.3f}")

    print(f"\nTop 5 least improved classes (representation may be deficient):")
    for entry in stats["top10_least_improved"][:5]:
        print(f"  {entry['ucf_class']:<25}  delta={entry['delta']:+.4f}  "
              f"probe={entry['linear_probe_acc']:.3f}  ssv2_conf={entry['ssv2_confidence']:.3f}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings, labels, df, class_features = load_data()

    print("\n--- Running 5-Fold Stratified Linear Probe ---")
    mean_acc, std_acc, per_class_acc = run_linear_probe(embeddings, labels, df)
    print(f"\nOverall: {mean_acc:.4f} ± {std_acc:.4f}")

    print("\n--- Computing Statistics ---")
    stats = compute_statistics(per_class_acc, class_features)

    print("\n--- Generating Plot ---")
    generate_plot(stats, mean_acc, std_acc)

    save_results(stats, mean_acc, std_acc)

    print("\nPhase 4 complete.")
    print("Results saved to results/linear_probe_results.json")
    print("Plot saved to results/plots/linear_probe.png")
    print("\nReady for Phase 3 Step 5: Final Report")

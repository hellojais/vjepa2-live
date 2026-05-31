# =============================================================================
# vjepa2_ucf101_eval.py — Phase 2: UCF-101 Out-of-Distribution Behavioral Study
# =============================================================================
# Goal: purely observational — no label mapping, no accuracy measurement.
# "What does V-JEPA 2 predict when shown UCF-101 videos it was never trained on,
#  and what patterns emerge in those predictions?"
# =============================================================================

# =============================================================================
# Section 1 — Imports and Config
# =============================================================================
import base64
import collections
import datetime
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from vjepa2_webcam import device, model, processor, run_inference

# Top-level constants
UCF101_ROOT          = Path("data/UCF-101")
SPLIT_FILE           = Path("data/ucfTrainTestlist/testlist01.txt")
NUM_FRAMES           = 16
RESULTS_DIR          = Path("results")
OUTPUT_CSV           = RESULTS_DIR / "ucf101_predictions.csv"
MAX_VIDEOS_PER_CLASS = 10


# =============================================================================
# Section 2 — Video Decoder
# =============================================================================
def load_frames(video_path: Path, num_frames: int = NUM_FRAMES) -> np.ndarray | None:
    """
    Load and uniformly sample num_frames from a video using torchcodec.
    Returns shape (num_frames, H, W, 3) uint8 RGB numpy array, or None on error.
    """
    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(video_path))
        total_frames = len(decoder)

        if total_frames == 0:
            print(f"[WARN] Zero frames in {video_path}")
            return None

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        frames = decoder.get_frames_at(indices=indices)

        # frames.data: (N, C, H, W) — may be float [0,1] or uint8 [0,255]
        result = []
        for i in range(len(indices)):
            frame_tensor = frames.data[i].permute(1, 2, 0).cpu()
            if frame_tensor.dtype == torch.float32:
                frame_np = (frame_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                frame_np = frame_tensor.numpy().astype(np.uint8)
            result.append(frame_np)

        return np.stack(result, axis=0)  # (num_frames, H, W, 3)

    except Exception as e:
        print(f"[ERROR] Failed to decode {video_path}: {e}")
        return None


# =============================================================================
# Section 3 — Single Video Inference
# =============================================================================
def infer_video(video_path: Path, ucf_class: str) -> dict | None:
    """
    Run inference on a single video.
    Returns a result dict or None on failure.
    """
    frames = load_frames(video_path)
    if frames is None:
        return None

    frame_deque = collections.deque(list(frames), maxlen=16)

    t0 = time.perf_counter()
    predictions = run_inference(frame_deque, processor, model, device)
    t1 = time.perf_counter()

    inference_time_ms = (t1 - t0) * 1000.0

    # Pad to 3 entries if fewer returned
    while len(predictions) < 3:
        predictions.append(("", 0.0))

    top1_label, top1_prob = predictions[0]
    top2_label, top2_prob = predictions[1]
    top3_label, top3_prob = predictions[2]

    # Entropy over top-3 probs as a confidence-spread measure
    probs = np.array([top1_prob, top2_prob, top3_prob], dtype=np.float64)
    probs_nonzero = probs[probs > 0]
    top1_entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero + 1e-10)))

    return {
        "video_path":       str(video_path),
        "ucf_class":        ucf_class,
        "top1_label":       top1_label,
        "top1_prob":        float(top1_prob),
        "top2_label":       top2_label,
        "top2_prob":        float(top2_prob),
        "top3_label":       top3_label,
        "top3_prob":        float(top3_prob),
        "inference_time_ms": inference_time_ms,
        "top1_entropy":     top1_entropy,
    }


# =============================================================================
# Section 4 — Batch Inference Loop
# =============================================================================
def run_batch(ucf101_root: Path, split_file: Path) -> pd.DataFrame:
    """
    Parse split file, run inference on all videos, return results as DataFrame.
    Saves incrementally to OUTPUT_CSV every 50 videos for crash recovery.
    """
    # Parse split file
    lines = split_file.read_text().strip().splitlines()
    entries: list[tuple[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Format may be "ClassName/v_....avi" or "ClassName/v_....avi 1"
        rel_path = line.split()[0]
        ucf_class = Path(rel_path).parts[0]
        entries.append((rel_path, ucf_class))

    # Group by class and apply MAX_VIDEOS_PER_CLASS cap
    class_groups: dict[str, list[str]] = collections.defaultdict(list)
    for rel_path, ucf_class in entries:
        class_groups[ucf_class].append(rel_path)

    capped_entries: list[tuple[str, str]] = []
    for ucf_class, paths in sorted(class_groups.items()):
        for rel_path in paths[:MAX_VIDEOS_PER_CLASS]:
            capped_entries.append((rel_path, ucf_class))

    total_videos  = len(capped_entries)
    total_classes = len(class_groups)
    print(f"\nTotal classes in split : {total_classes}")
    print(f"Videos to process      : {total_videos}  (cap={MAX_VIDEOS_PER_CLASS}/class)\n")

    results:      list[dict] = []
    error_count:  int        = 0
    csv_written:  bool       = False

    for idx, (rel_path, ucf_class) in enumerate(
        tqdm(capped_entries, desc="Inferring"), start=1
    ):
        video_path = ucf101_root / rel_path
        result = infer_video(video_path, ucf_class)

        if result is None:
            error_count += 1
        else:
            results.append(result)

        # Incremental save + running stats every 50 videos
        if idx % 50 == 0 and results:
            chunk = pd.DataFrame(results)
            if not csv_written:
                chunk.to_csv(OUTPUT_CSV, index=False)
                csv_written = True
            else:
                chunk.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
                results = []  # flush buffer after append

            mean_conf = chunk["top1_prob"].mean()
            print(
                f"\n[{idx}/{total_videos}] "
                f"Processed: {idx - error_count} | "
                f"Errors: {error_count} | "
                f"Mean top1_prob: {mean_conf:.3f}"
            )

    # Final flush for any remaining results not yet saved
    if results:
        final_df = pd.DataFrame(results)
        if not csv_written:
            final_df.to_csv(OUTPUT_CSV, index=False)
        else:
            final_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    print(f"\nAll results saved to {OUTPUT_CSV}")

    # Reload full CSV to return complete DataFrame
    if OUTPUT_CSV.exists():
        return pd.read_csv(OUTPUT_CSV)
    return pd.DataFrame()


# =============================================================================
# Section 5 — Pattern Classifier
# =============================================================================
def classify_behavior(group: pd.DataFrame) -> str:
    """Classify prediction behavior for one UCF class into one of three categories."""
    if len(group) == 0:
        return "LOW_CONF_SCATTERED"

    mean_confidence = group["top1_prob"].mean()
    top_label_count = group["top1_label"].value_counts().iloc[0]
    consistency     = top_label_count / len(group)

    if mean_confidence >= 0.4 and consistency >= 0.6:
        return "HIGH_CONF_CONSISTENT"
    elif mean_confidence >= 0.4 and consistency < 0.6:
        return "HIGH_CONF_INCONSISTENT"
    else:
        return "LOW_CONF_SCATTERED"


def build_pattern_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-class pattern report.
    Saves results/pattern_report.json and returns a DataFrame.
    """
    if df.empty or "ucf_class" not in df.columns:
        print("[WARN] No results to build pattern report from.")
        return pd.DataFrame()

    rows = []
    for ucf_class, group in df.groupby("ucf_class"):
        behavior      = classify_behavior(group)
        vc            = group["top1_label"].value_counts()
        most_common   = vc.index[0]  if len(vc) > 0 else ""
        common_count  = int(vc.iloc[0]) if len(vc) > 0 else 0
        unique_preds  = int(group["top1_label"].nunique())

        rows.append({
            "ucf_class":              ucf_class,
            "behavior":               behavior,
            "mean_top1_prob":         float(group["top1_prob"].mean()),
            "most_common_ssv2":       most_common,
            "most_common_ssv2_count": common_count,
            "unique_ssv2_predictions": unique_preds,
            "mean_entropy":           float(group["top1_entropy"].mean()),
            "video_count":            len(group),
        })

    pattern_df = pd.DataFrame(rows)

    behavior_order = {
        "HIGH_CONF_CONSISTENT":   0,
        "HIGH_CONF_INCONSISTENT": 1,
        "LOW_CONF_SCATTERED":     2,
    }
    pattern_df["_rank"] = pattern_df["behavior"].map(behavior_order)
    pattern_df = (
        pattern_df
        .sort_values(["_rank", "mean_top1_prob"], ascending=[True, False])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    json_path = RESULTS_DIR / "pattern_report.json"
    pattern_df.to_json(json_path, orient="records", indent=2)
    print(f"Pattern report saved to {json_path}")
    return pattern_df


# =============================================================================
# Section 6 — Visualizations
# =============================================================================
BEHAVIOR_PALETTE = {
    "HIGH_CONF_CONSISTENT":   "#22c55e",
    "HIGH_CONF_INCONSISTENT": "#f59e0b",
    "LOW_CONF_SCATTERED":     "#ef4444",
}


def generate_plots(df: pd.DataFrame, pattern_df: pd.DataFrame) -> None:
    """Generate and save all 4 analysis plots to RESULTS_DIR."""
    sns.set_style("darkgrid")

    # ── Plot 1: Confidence Distribution by Behavior ──────────────────────────
    behavior_map = pattern_df.set_index("ucf_class")["behavior"].to_dict()
    df_plot = df.copy()
    df_plot["behavior"] = df_plot["ucf_class"].map(behavior_map)

    behaviors_present = [
        b for b in ["HIGH_CONF_CONSISTENT", "HIGH_CONF_INCONSISTENT", "LOW_CONF_SCATTERED"]
        if b in df_plot["behavior"].values
    ]
    palette = {b: BEHAVIOR_PALETTE[b] for b in behaviors_present}

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=df_plot,
        x="behavior",
        y="top1_prob",
        palette=palette,
        order=behaviors_present,
        ax=ax,
    )
    ax.set_title("Top-1 Confidence Distribution by Prediction Behavior", fontsize=14)
    ax.set_xlabel("Behavior Category")
    ax.set_ylabel("Top-1 Probability")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confidence_by_behavior.png", dpi=120)
    plt.close(fig)
    print("Saved confidence_by_behavior.png")

    # ── Plot 2: Top SSv2 Predictions Heatmap ─────────────────────────────────
    top20_classes = pattern_df.nlargest(20, "mean_top1_prob")["ucf_class"].tolist()
    df_top20 = df[df["ucf_class"].isin(top20_classes)]
    pivot = (
        df_top20
        .groupby(["ucf_class", "top1_label"])
        .size()
        .unstack(fill_value=0)
    )
    # Keep top-10 most predicted SSv2 labels for readability
    top_ssv2 = pivot.sum(axis=0).nlargest(10).index
    pivot = pivot.reindex(columns=top_ssv2, fill_value=0)

    fig, ax = plt.subplots(figsize=(24, 10))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        annot=True,
        fmt="d",
        cbar_kws={"label": "Video Count"},
    )
    ax.set_title(
        "Most Common SSv2 Predictions per UCF Class (Top 20 by Confidence)",
        fontsize=13,
    )
    ax.set_xlabel("SSv2 Predicted Label")
    ax.set_ylabel("UCF-101 Class")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "ssv2_prediction_heatmap.png", dpi=100)
    plt.close(fig)
    print("Saved ssv2_prediction_heatmap.png")

    # ── Plot 3: Behavior Distribution Pie Chart ───────────────────────────────
    behavior_counts = pattern_df["behavior"].value_counts()
    colors = [BEHAVIOR_PALETTE.get(b, "#888") for b in behavior_counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        behavior_counts.values,
        labels=behavior_counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
    )
    ax.set_title("UCF-101 Class Behavior Distribution", fontsize=14)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "behavior_distribution.png", dpi=120)
    plt.close(fig)
    print("Saved behavior_distribution.png")

    # ── Plot 4: Entropy vs Confidence Scatter ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))

    for behavior, grp in pattern_df.groupby("behavior"):
        color = BEHAVIOR_PALETTE.get(behavior, "#888")
        ax.scatter(
            grp["mean_entropy"],
            grp["mean_top1_prob"],
            label=behavior,
            color=color,
            alpha=0.75,
            s=60,
        )

    # Annotate the 5 highest-confidence points
    for _, row in pattern_df.nlargest(5, "mean_top1_prob").iterrows():
        ax.annotate(
            row["ucf_class"],
            (row["mean_entropy"], row["mean_top1_prob"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_title("Prediction Entropy vs Confidence per UCF Class", fontsize=14)
    ax.set_xlabel("Mean Top-1 Entropy")
    ax.set_ylabel("Mean Top-1 Probability")
    ax.legend(title="Behavior")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "entropy_vs_confidence.png", dpi=120)
    plt.close(fig)
    print("Saved entropy_vs_confidence.png")


# =============================================================================
# Section 7 — HTML Summary Report
# =============================================================================
def _img_to_b64(img_path: Path) -> str:
    """Read an image file and return a base64-encoded string."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_html_report(df: pd.DataFrame, pattern_df: pd.DataFrame) -> None:
    """
    Generate a single self-contained dark-themed HTML report.
    All plots are embedded as inline base64 images — no external dependencies.
    """
    total_videos  = len(df)
    total_classes = df["ucf_class"].nunique()
    date_run      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    ROW_BG = {
        "HIGH_CONF_CONSISTENT":   "rgba(34,197,94,0.07)",
        "HIGH_CONF_INCONSISTENT": "rgba(245,158,11,0.07)",
        "LOW_CONF_SCATTERED":     "rgba(239,68,68,0.07)",
    }

    # ── Behavior summary cards ────────────────────────────────────────────────
    behavior_counts = pattern_df["behavior"].value_counts().to_dict()
    cards_html = ""
    for behavior in ["HIGH_CONF_CONSISTENT", "HIGH_CONF_INCONSISTENT", "LOW_CONF_SCATTERED"]:
        count  = behavior_counts.get(behavior, 0)
        color  = BEHAVIOR_PALETTE[behavior]
        top5   = (
            pattern_df[pattern_df["behavior"] == behavior]
            .nlargest(5, "mean_top1_prob")["ucf_class"]
            .tolist()
        )
        items  = "".join(f"<li>{c}</li>" for c in top5)
        label  = behavior.replace("_", " ")
        cards_html += f"""
        <div class="card" style="border-top:3px solid {color};">
          <div class="card-title" style="color:{color};">{label}</div>
          <div class="card-count">{count}</div>
          <div class="card-sub">classes</div>
          <ul class="card-list">{items}</ul>
        </div>"""

    # ── Embedded plots ────────────────────────────────────────────────────────
    plot_specs = [
        ("confidence_by_behavior.png",  "Confidence Distribution by Behavior"),
        ("ssv2_prediction_heatmap.png", "SSv2 Prediction Heatmap"),
        ("behavior_distribution.png",   "Behavior Distribution"),
        ("entropy_vs_confidence.png",   "Entropy vs Confidence"),
    ]
    plots_html = ""
    for fname, title in plot_specs:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            b64 = _img_to_b64(fpath)
            plots_html += f"""
            <div class="plot-block">
              <h3>{title}</h3>
              <img src="data:image/png;base64,{b64}" alt="{title}" />
            </div>"""

    # ── Full pattern table ────────────────────────────────────────────────────
    table_rows = ""
    for _, row in pattern_df.iterrows():
        bg    = ROW_BG.get(row["behavior"], "")
        color = BEHAVIOR_PALETTE.get(row["behavior"], "#aaa")
        label = row["behavior"].replace("_", " ")
        table_rows += f"""
        <tr style="background:{bg};">
          <td>{row['ucf_class']}</td>
          <td style="color:{color};font-weight:600;">{label}</td>
          <td>{row['mean_top1_prob']:.3f}</td>
          <td style="font-size:11px;">{row['most_common_ssv2']}</td>
          <td>{row['most_common_ssv2_count']}</td>
          <td>{row['unique_ssv2_predictions']}</td>
          <td>{row['mean_entropy']:.3f}</td>
        </tr>"""

    # ── Phase 3 candidates ────────────────────────────────────────────────────
    def _candidate_rows(sub: pd.DataFrame) -> str:
        return "".join(
            f"<tr><td>{r['ucf_class']}</td>"
            f"<td>{r['mean_top1_prob']:.3f}</td>"
            f"<td style='font-size:11px;'>{r['most_common_ssv2']}</td></tr>"
            for _, r in sub.iterrows()
        )

    top_consistent   = (
        pattern_df[pattern_df["behavior"] == "HIGH_CONF_CONSISTENT"]
        .nlargest(5, "mean_top1_prob")[["ucf_class", "mean_top1_prob", "most_common_ssv2"]]
    )
    top_inconsistent = (
        pattern_df[pattern_df["behavior"] == "HIGH_CONF_INCONSISTENT"]
        .nlargest(5, "mean_top1_prob")[["ucf_class", "mean_top1_prob", "most_common_ssv2"]]
    )

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>V-JEPA 2 — Phase 2 Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #07070f;
      color: #d4d4d8;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI",
                   system-ui, sans-serif;
      padding-bottom: 60px;
    }}
    header {{
      background: #0d0d18;
      padding: 28px 40px;
      border-bottom: 1px solid #18181f;
    }}
    header h1 {{ font-size: 22px; font-weight: 700; color: #e4e4e7; letter-spacing: -0.4px; }}
    header h1 span {{ color: #6366f1; }}
    .subtitle {{ font-size: 13px; color: #52525b; margin-top: 5px; }}
    .meta {{ font-size: 12px; color: #3f3f5a; margin-top: 8px; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 0 40px; }}
    section {{ margin-top: 44px; }}
    h2 {{
      font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px;
      color: #3f3f5a; margin-bottom: 18px; padding-bottom: 8px;
      border-bottom: 1px solid #18181f;
    }}
    h3 {{ font-size: 13px; font-weight: 500; color: #71717a; margin-bottom: 10px; }}
    /* cards */
    .cards {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .card {{
      flex: 1; min-width: 220px;
      background: #0d0d18; border: 1px solid #18181f; border-radius: 12px;
      padding: 20px;
    }}
    .card-title {{
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.8px; margin-bottom: 8px;
    }}
    .card-count {{ font-size: 32px; font-weight: 700; color: #e4e4e7; line-height: 1; }}
    .card-sub {{ font-size: 11px; color: #52525b; margin-bottom: 12px; }}
    .card-list {{ list-style: none; font-size: 12px; color: #71717a; line-height: 1.9; }}
    /* plots */
    .plots {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .plot-block {{
      background: #0d0d18; border: 1px solid #18181f;
      border-radius: 12px; padding: 20px;
    }}
    .plot-block img {{ width: 100%; border-radius: 6px; margin-top: 8px; }}
    /* table */
    .table-wrap {{
      overflow-x: auto;
      background: #0a0a14;
      border: 1px solid #18181f;
      border-radius: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{
      background: #0d0d18; color: #52525b; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.5px;
      padding: 10px 14px; text-align: left;
      border-bottom: 1px solid #18181f;
      position: sticky; top: 0;
    }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #111118; color: #c4c4c8; }}
    tr:hover td {{ background: rgba(99,102,241,0.05); }}
    /* phase 3 */
    .phase3-box {{
      background: #0d0d18;
      border: 1px solid rgba(99,102,241,0.3);
      border-radius: 12px; padding: 24px;
    }}
    .p3-tables {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 18px; }}
    .p3-table-wrap {{ flex: 1; min-width: 280px; }}
    .p3-table-wrap h3 {{ color: #a5b4fc; }}
    .phase3-note {{
      font-size: 12px; color: #71717a; margin-top: 18px;
      padding: 14px; background: #111118; border-radius: 8px;
      border-left: 3px solid #6366f1; line-height: 1.7;
    }}
  </style>
</head>
<body>

<header>
  <div class="container">
    <h1>V-JEPA <span>2</span> &mdash; Phase 2: Prediction Pattern Analysis</h1>
    <div class="subtitle">UCF-101 out-of-distribution behavioral study</div>
    <div class="meta">
      Total videos: {total_videos} &nbsp;&middot;&nbsp;
      Total classes: {total_classes} &nbsp;&middot;&nbsp;
      Run: {date_run}
    </div>
  </div>
</header>

<div class="container">

  <section>
    <h2>Behavior Summary</h2>
    <div class="cards">{cards_html}</div>
  </section>

  <section>
    <h2>Visualizations</h2>
    <div class="plots">{plots_html}</div>
  </section>

  <section>
    <h2>Full Pattern Table</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>UCF Class</th>
            <th>Behavior</th>
            <th>Mean Confidence</th>
            <th>Most Common SSv2 Prediction</th>
            <th>Count</th>
            <th>Unique SSv2</th>
            <th>Mean Entropy</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Phase 3 Candidates</h2>
    <div class="phase3-box">
      <div class="p3-tables">
        <div class="p3-table-wrap">
          <h3>Top 5 &mdash; HIGH CONF CONSISTENT</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>UCF Class</th><th>Confidence</th><th>Predicted SSv2</th></tr>
              </thead>
              <tbody>{_candidate_rows(top_consistent)}</tbody>
            </table>
          </div>
        </div>
        <div class="p3-table-wrap">
          <h3>Top 5 &mdash; HIGH CONF INCONSISTENT</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>UCF Class</th><th>Confidence</th><th>Top Predicted SSv2</th></tr>
              </thead>
              <tbody>{_candidate_rows(top_inconsistent)}</tbody>
            </table>
          </div>
        </div>
      </div>
      <div class="phase3-note">
        These classes are recommended starting points for Phase 3 embedding geometry analysis.<br>
        <strong>HIGH_CONF_CONSISTENT</strong> classes show strong transferable signal —
        the encoder has learned something real about these motions.<br>
        <strong>HIGH_CONF_INCONSISTENT</strong> classes reveal the most interesting failure
        modes — high-confidence predictions that shift per video.
      </div>
    </div>
  </section>

</div>
</body>
</html>"""

    report_path = RESULTS_DIR / "phase2_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"HTML report saved to {report_path}")


# =============================================================================
# Section 8 — Entry Point
# =============================================================================
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    instructions = """
    UCF-101 Setup Instructions:
    Dataset : https://www.crcv.ucf.edu/data/UCF101/UCF101.tar.gz
    Splits  : https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
    Extract dataset to : data/UCF-101/
    Extract splits to  : data/ucfTrainTestlist/
    """
    print(instructions)
    Path("data").mkdir(exist_ok=True)
    (Path("data") / "README.txt").write_text(instructions)

    assert UCF101_ROOT.exists(), (
        f"UCF-101 not found at {UCF101_ROOT}. See data/README.txt"
    )
    assert SPLIT_FILE.exists(), (
        f"Split file not found at {SPLIT_FILE}. See data/README.txt"
    )

    df          = run_batch(UCF101_ROOT, SPLIT_FILE)
    pattern_df  = build_pattern_report(df)
    generate_plots(df, pattern_df)
    generate_html_report(df, pattern_df)
    print("Phase 2 complete. Open results/phase2_report.html in your browser.")

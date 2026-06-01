"""Phase 3 Step 5 — Final Report Generation.

Loads all results from disk and produces:
  results/vjepa2_live_report.html  — self-contained HTML report
  results/vjepa2_live_report.md    — Markdown summary
  README.md                        — GitHub README

No model loading, no recomputation — pure file I/O.
"""

from pathlib import Path
import json
import base64
import datetime
from collections import Counter
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PLOTS_DIR = Path("results/plots")
RESULTS_DIR = Path("results")
REPORT_OUTPUT = Path("results/vjepa2_live_report.html")

BEHAVIOR_COLORS = {
    "HIGH_CONF_CONSISTENT": "#2ecc71",
    "HIGH_CONF_INCONSISTENT": "#f39c12",
    "LOW_CONF_SCATTERED": "#e74c3c",
}

BLIND_SPOT_WHY = {
    "PoleVault":      "Body inversion — no SSv2 hand-object contact",
    "SalsaSpin":      "Rotation without hand-object contact",
    "Surfing":        "Lateral body sweep misread as wiping",
    "WritingOnBoard": "Arm arc misread as dispersal / sprinkling",
}

CENTRAL_FINDING = (
    "V-JEPA 2\u2019s self-supervised backbone achieves 98.2% linear probe "
    "accuracy on UCF-101 \u2014 a dataset it was never trained on \u2014 while its "
    "SSv2 classification head produces confident but semantically wrong "
    "predictions for 42% of the same classes. The bottleneck is not the "
    "representation. It is the vocabulary."
)


# ---------------------------------------------------------------------------
# Section 2 — Data Loading
# ---------------------------------------------------------------------------

def img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_all_results() -> dict:
    # pattern_report.json — list; field is mean_top1_prob (Fix 4)
    pattern_report = json.loads(Path("results/pattern_report.json").read_text())
    pr_by_class = {r["ucf_class"]: r for r in pattern_report}

    lp = json.loads(Path("results/linear_probe_results.json").read_text())
    step3 = json.loads(Path("results/step3_findings.json").read_text())
    step4 = json.loads(Path("results/step4_findings.json").read_text())
    # class_features.csv uses top1_prob_mean (Fix 4 — never swap)
    class_features = pd.read_csv(Path("results/class_features.csv"))

    # Fix 1: compute delta by joining pattern_report + linear_probe at load time
    per_class_delta = {
        cls: lp["per_class_accuracy"][cls] - pr_by_class[cls]["mean_top1_prob"]
        for cls in lp["per_class_accuracy"]
        if cls in pr_by_class
    }

    # Fix 3: top 10 Phase 2 mappings by mean_top1_prob descending
    top10_mappings = sorted(
        pattern_report, key=lambda r: r["mean_top1_prob"], reverse=True
    )[:10]

    plot_files = [
        "behavior_clusters.png", "ssv2_label_regions.png", "ucf_class_clusters.png",
        "confidence_gradient.png", "blind_spot_analysis.png", "hitting_cluster.png",
        "behavior_geometry.png", "hypothesis_a_variance.png", "hypothesis_b_distance.png",
        "hypothesis_c_similarity.png", "feature_summary.png", "linear_probe.png",
    ]
    plots = {}
    for name in plot_files:
        p = PLOTS_DIR / name
        if p.exists():
            plots[name.replace(".png", "")] = img_to_base64(p)

    beh_counts = Counter(r["behavior"] for r in pattern_report)

    return {
        "pattern_report": pattern_report,
        "pr_by_class": pr_by_class,
        "linear_probe": lp,
        "per_class_delta": per_class_delta,
        "class_features": class_features,
        "step3": step3,
        "step4": step4,
        "top10_mappings": top10_mappings,
        "beh_counts": beh_counts,
        "plots": plots,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ---------------------------------------------------------------------------
# CSS and JS helpers (pure strings — no f-string, so { } are literal)
# ---------------------------------------------------------------------------

def _css() -> str:
    return """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: #0d1117; color: #e6edf3;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
  line-height: 1.7; font-size: 16px;
}
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: #161b22; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; color: #79c0ff; }
h1 { font-size: 2.4rem; font-weight: 700; color: #e6edf3; margin-bottom: 0.5rem; }
h2 { font-size: 1.6rem; font-weight: 600; color: #58a6ff; margin: 2rem 0 1rem; border-bottom: 1px solid #30363d; padding-bottom: 0.4rem; }
h3 { font-size: 1.2rem; font-weight: 600; color: #cdd9e5; margin: 1.5rem 0 0.75rem; }
p  { margin-bottom: 1rem; color: #cdd9e5; }

/* Nav */
nav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 100;
  background: #161b22cc; backdrop-filter: blur(8px);
  border-bottom: 1px solid #30363d;
  padding: 0.6rem 2rem; display: flex; gap: 1.5rem; align-items: center;
  flex-wrap: wrap;
}
nav .nav-brand { font-weight: 700; color: #e6edf3; margin-right: auto; }
nav a { color: #8b949e; font-size: 0.9rem; }
nav a:hover { color: #58a6ff; text-decoration: none; }

/* Layout */
.page { max-width: 1200px; margin: 0 auto; padding: 5rem 2rem 4rem; }
section { margin-bottom: 4rem; }

/* Stat cards */
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 2rem 0; }
.stat-card {
  background: #161b22; border: 1px solid #30363d; border-radius: 10px;
  padding: 1.5rem; text-align: center;
}
.stat-number { font-size: 3rem; font-weight: 700; color: #58a6ff; display: block; }
.stat-label  { font-size: 0.9rem; color: #8b949e; margin-top: 0.3rem; }

/* Pull quote */
.pull-quote {
  border-left: 4px solid #58a6ff; background: #161b22;
  padding: 1.5rem 2rem; border-radius: 0 8px 8px 0;
  font-size: 1.15rem; color: #e6edf3; font-style: italic;
  margin: 2rem 0;
}
.pull-quote strong { color: #58a6ff; font-style: normal; }

/* Callout box */
.callout {
  background: #1c2128; border: 1px solid #388bfd44;
  border-left: 4px solid #58a6ff; border-radius: 8px;
  padding: 1.2rem 1.5rem; margin: 1.5rem 0;
}
.callout.green  { border-color: #2ecc7144; border-left-color: #2ecc71; }
.callout.yellow { border-color: #f39c1244; border-left-color: #f39c12; }
.callout.red    { border-color: #e74c3c44; border-left-color: #e74c3c; }

/* Behavior cards */
.beh-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin: 1.5rem 0; }
.beh-card {
  border-radius: 10px; padding: 1.2rem 1.5rem;
  border: 1px solid;
}
.beh-card .beh-count { font-size: 2rem; font-weight: 700; display: block; }
.beh-card .beh-name  { font-size: 0.85rem; margin-top: 0.2rem; opacity: 0.85; }

/* Tables */
.table-wrap { overflow-x: auto; margin: 1.5rem 0; border-radius: 8px; border: 1px solid #30363d; }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.data-table thead th {
  background: #161b22; color: #8b949e; padding: 0.75rem 1rem;
  text-align: left; border-bottom: 1px solid #30363d;
  cursor: pointer; user-select: none; white-space: nowrap;
}
.data-table thead th:hover { color: #58a6ff; }
.data-table tbody td { padding: 0.65rem 1rem; border-bottom: 1px solid #21262d; }
.data-table tbody tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover { background: #1c2128 !important; }

/* Badges */
.beh-badge {
  display: inline-block; padding: 2px 8px; border-radius: 20px;
  font-size: 0.75rem; font-weight: 500; white-space: nowrap;
}

/* Plots */
.plot-img { width: 100%; border-radius: 8px; border: 1px solid #30363d; margin: 1rem 0; display: block; }
.plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }
.plot-grid .plot-img { margin: 0; }
.missing-plot { color: #8b949e; font-style: italic; padding: 1rem; }

/* Metadata bar */
.meta-bar {
  display: flex; gap: 2rem; flex-wrap: wrap;
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 0.75rem 1.5rem; margin: 1.5rem 0; font-size: 0.875rem; color: #8b949e;
}
.meta-bar span strong { color: #cdd9e5; }

/* Phase header */
.section-header {
  background: linear-gradient(135deg, #161b22, #1c2128);
  border: 1px solid #30363d; border-radius: 12px;
  padding: 3rem 2.5rem; text-align: center; margin-bottom: 3rem;
}
.section-header .subtitle { color: #8b949e; font-size: 1.1rem; margin-top: 0.5rem; }

/* Conclusions */
.conclusion-list { list-style: none; counter-reset: conc; }
.conclusion-list li {
  counter-increment: conc; position: relative;
  padding: 1rem 1rem 1rem 3.5rem; margin-bottom: 0.75rem;
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
}
.conclusion-list li::before {
  content: counter(conc); position: absolute; left: 1rem; top: 1rem;
  width: 1.8rem; height: 1.8rem; background: #58a6ff22; color: #58a6ff;
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.9rem;
}

/* Responsive */
@media (max-width: 768px) {
  .stat-grid, .beh-grid, .plot-grid { grid-template-columns: 1fr; }
  nav { padding: 0.5rem 1rem; gap: 0.75rem; }
  .page { padding: 4rem 1rem 3rem; }
}

/* Print */
@media print {
  nav { display: none; }
  .page { padding-top: 1rem; }
  .plot-img { break-inside: avoid; }
}
"""


def _js() -> str:
    return """
function sortTable(tableId, colIdx) {
    const tbl = document.getElementById(tableId);
    if (!tbl) return;
    const tbody = tbl.tBodies[0];
    const rows = Array.from(tbody.rows);
    const th = tbl.tHead.rows[0].cells[colIdx];
    const asc = th.dataset.dir !== 'asc';

    rows.sort((a, b) => {
        const av = a.cells[colIdx].textContent.replace(/[+↑↓%]/g, '').trim();
        const bv = b.cells[colIdx].textContent.replace(/[+↑↓%]/g, '').trim();
        const na = parseFloat(av);
        const nb = parseFloat(bv);
        if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    });

    Array.from(tbl.tHead.rows[0].cells).forEach(c => {
        c.dataset.dir = '';
        c.textContent = c.textContent.replace(/\\s*[↑↓]$/, '').trim();
    });
    th.dataset.dir = asc ? 'asc' : 'desc';
    th.textContent += asc ? ' ↑' : ' ↓';
    rows.forEach(r => tbody.appendChild(r));
}
"""


# ---------------------------------------------------------------------------
# Section 3 — HTML Generation
# ---------------------------------------------------------------------------

def _img(plots: dict, key: str, alt: str = "") -> str:
    if key in plots:
        return f'<img src="data:image/png;base64,{plots[key]}" alt="{alt}" class="plot-img">'
    return f'<p class="missing-plot">Plot not available: {key}.png</p>'


def generate_html(data: dict) -> str:
    css = _css()
    js = _js()
    lp = data["linear_probe"]
    s3 = data["step3"]
    s4 = data["step4"]
    plots = data["plots"]
    beh_counts = data["beh_counts"]
    gen_date = data["generated_at"]

    # ---------- nav ----------
    nav = """
<nav>
  <span class="nav-brand">V-JEPA 2 Live</span>
  <a href="#executive-summary">Summary</a>
  <a href="#study-design">Design</a>
  <a href="#phase2">Phase 2</a>
  <a href="#phase3">Phase 3</a>
  <a href="#phase4">Phase 4</a>
  <a href="#conclusions">Conclusions</a>
  <a href="#appendix">Appendix</a>
</nav>"""

    # ---------- header ----------
    header = f"""
<div class="section-header">
  <h1>V-JEPA 2 Live</h1>
  <p class="subtitle">Probing Self-Supervised Video Representations &mdash; A systematic out-of-distribution study on UCF-101</p>
  <div class="meta-bar" style="margin-top:1.5rem;justify-content:center">
    <span><strong>Date:</strong> {gen_date}</span>
    <span><strong>Model:</strong> qubvel-hf/vjepa2-vitl-fpc16-256-ssv2</span>
    <span><strong>Dataset:</strong> UCF-101 (1059 clips)</span>
    <span><strong>Hardware:</strong> MacBook Pro M5 Max &middot; MPS</span>
  </div>
</div>"""

    # ---------- executive summary ----------
    exec_summary = f"""
<section id="executive-summary">
  <h2>Executive Summary</h2>
  <div class="pull-quote">
    <strong>{CENTRAL_FINDING}</strong>
  </div>
  <div class="stat-grid">
    <div class="stat-card">
      <span class="stat-number">98.2%</span>
      <div class="stat-label">Linear Probe Accuracy on UCF-101</div>
    </div>
    <div class="stat-card">
      <span class="stat-number">42%</span>
      <div class="stat-label">UCF-101 classes the SSv2 head fails on</div>
    </div>
    <div class="stat-card">
      <span class="stat-number">0.93</span>
      <div class="stat-label">Backbone same-class cosine similarity</div>
    </div>
  </div>
  <p>
    V-JEPA 2 is a ViT-L (325M parameter) video transformer pre-trained with a joint-embedding predictive
    architecture (JEPA) and fine-tuned for classification on Something-Something v2 (SSv2) &mdash;
    174 hand-object interaction classes. This project applied the model <em>frozen and zero-shot</em>
    to UCF-101 (101 action classes it was never trained on), then systematically probed its representations
    across five phases to understand what its world model actually encodes.
  </p>
  <p>
    The central finding: the backbone generalises almost perfectly to unseen action categories,
    achieving 98.2% top-1 accuracy with a simple linear classifier on the frozen embeddings.
    All four classes the SSv2 head mapped to nonsensical physics primitives (the &ldquo;blind spots&rdquo;)
    achieve 100% linear probe accuracy. The SSv2 classification head &mdash; not the backbone &mdash;
    is entirely responsible for the model&rsquo;s poor zero-shot UCF-101 behaviour.
  </p>
  <div class="callout">
    &ldquo;The bottleneck is not the representation. It is the vocabulary.&rdquo;
  </div>
</section>"""

    # ---------- study design ----------
    study_design = f"""
<section id="study-design">
  <h2>Study Design</h2>
  <div class="meta-bar">
    <span><strong>Model:</strong> V-JEPA 2 ViT-L, 325M params, SSv2-fine-tuned, frozen throughout</span>
    <span><strong>Dataset:</strong> UCF-101, 101 classes, ~10 clips/class</span>
    <span><strong>Hardware:</strong> MacBook Pro M5 Max, PyTorch 2.12 MPS, no CUDA</span>
  </div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Phase</th><th>Goal</th><th>Key Output</th></tr></thead>
      <tbody>
        <tr><td>Phase 1</td><td>Validate V-JEPA 2 on Apple Silicon MPS</td><td>~10fps live inference, importable module</td></tr>
        <tr><td>Phase 2</td><td>Zero-shot UCF-101 batch evaluation</td><td>101 classes categorised into 3 behaviour groups</td></tr>
        <tr><td>Phase 3 &ndash; Step 1</td><td>Extract backbone embeddings</td><td>(1059, 1024) float32 embedding matrix</td></tr>
        <tr><td>Phase 3 &ndash; Step 2</td><td>t-SNE + UMAP visualisation</td><td>4 embedding-space plots</td></tr>
        <tr><td>Phase 3 &ndash; Step 3</td><td>Nearest-neighbour analysis</td><td>Blind spots, hitting absorption, geometry&ndash;confidence correlation</td></tr>
        <tr><td>Phase 3 &ndash; Step 4</td><td>Hypothesis testing (3 proxies)</td><td>Temporal dynamics partially explains confidence; tight-cluster paradox identified</td></tr>
        <tr><td>Phase 4</td><td>Linear probe experiment</td><td>98.2% accuracy &mdash; SSv2 head is the bottleneck</td></tr>
      </tbody>
    </table>
  </div>
</section>"""

    # ---------- phase 2 ----------
    hcc = beh_counts.get("HIGH_CONF_CONSISTENT", 0)
    hci = beh_counts.get("HIGH_CONF_INCONSISTENT", 0)
    lcs = beh_counts.get("LOW_CONF_SCATTERED", 0)

    top10_rows = ""
    for r in data["top10_mappings"]:
        beh = r["behavior"]
        c = BEHAVIOR_COLORS.get(beh, "#58a6ff")
        top10_rows += f"""
        <tr>
          <td>{r["ucf_class"]}</td>
          <td><span class="beh-badge" style="background:{c}22;color:{c};border:1px solid {c}44">{beh}</span></td>
          <td>{r["most_common_ssv2"]}</td>
          <td>{r["mean_top1_prob"]:.3f}</td>
        </tr>"""

    phase2 = f"""
<section id="phase2">
  <h2>Phase 2 &mdash; Zero-Shot Behaviour Categorisation</h2>
  <p>
    1059 UCF-101 videos were run zero-shot through the SSv2 classification head. Each class was categorised
    by its prediction confidence and label consistency across ~10 videos.
  </p>
  <div class="beh-grid">
    <div class="beh-card" style="background:#2ecc7111;border-color:#2ecc7144;color:#2ecc71">
      <span class="beh-count">{hcc}</span>
      <div class="beh-name">HIGH_CONF_CONSISTENT</div>
      <p style="margin-top:0.5rem;font-size:0.85rem;color:#cdd9e5">
        Model maps confidently to one SSv2 concept across all videos of this class.
        Often semantically wrong but internally consistent.
      </p>
    </div>
    <div class="beh-card" style="background:#f39c1211;border-color:#f39c1244;color:#f39c12">
      <span class="beh-count">{hci}</span>
      <div class="beh-name">HIGH_CONF_INCONSISTENT</div>
      <p style="margin-top:0.5rem;font-size:0.85rem;color:#cdd9e5">
        High confidence but different SSv2 labels across videos of the same class.
        The backbone sees variation the head cannot resolve consistently.
      </p>
    </div>
    <div class="beh-card" style="background:#e74c3c11;border-color:#e74c3c44;color:#e74c3c">
      <span class="beh-count">{lcs}</span>
      <div class="beh-name">LOW_CONF_SCATTERED</div>
      <p style="margin-top:0.5rem;font-size:0.85rem;color:#cdd9e5">
        Low confidence, no dominant SSv2 label. SSv2 vocabulary has no good match
        for these action categories.
      </p>
    </div>
  </div>
  <h3>Top 10 Most Confidently Mapped Classes (SSv2 confidence descending)</h3>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>UCF Class</th><th>Behaviour</th><th>Most Common SSv2 Mapping</th><th>SSv2 Confidence</th></tr></thead>
      <tbody>{top10_rows}</tbody>
    </table>
  </div>
  <h3>Embedding Space: SSv2 Label Regions</h3>
  {_img(plots, "ssv2_label_regions", "SSv2 label regions in embedding space")}
</section>"""

    # ---------- phase 3 ----------
    # Blind spot table
    blind_rows = ""
    for cls, bs in s3["blind_spots"].items():
        nearest = bs["nearest_ssv2_concepts"][0][0] if bs.get("nearest_ssv2_concepts") else "N/A"
        why = BLIND_SPOT_WHY.get(cls, "No SSv2 vocabulary for this motion type")
        blind_rows += f"""
        <tr><td>{cls}</td><td>{nearest}</td><td>{why}</td></tr>"""

    # Hypothesis table
    hyp_rows = ""
    for key, label, proxy in [
        ("hypothesis_a", "Temporal Dynamics",  "intra_class_variance"),
        ("hypothesis_b", "Spatial Scale",       "distance_from_ssv2_center"),
        ("hypothesis_c", "Cluster Tightness",   "within_class_similarity"),
    ]:
        h = s4[key]
        vc = {"SUPPORTED": "#2ecc71", "PARTIALLY SUPPORTED": "#f39c12", "REJECTED": "#e74c3c"}.get(h["verdict"], "#58a6ff")
        hyp_rows += f"""
        <tr>
          <td>{label}</td>
          <td><code>{proxy}</code></td>
          <td>{h["pearson_r"]:+.3f}</td>
          <td>{h["pearson_p"]:.4f}</td>
          <td>{h["anova_f"]:.3f}</td>
          <td>{h["anova_p"]:.4f}</td>
          <td><span style="color:{vc};font-weight:600">{h["verdict"]}</span></td>
        </tr>"""

    hitting = s3["hitting_cluster"]
    tcp_classes = len(s4.get("classes_unexplained", []))

    phase3 = f"""
<section id="phase3">
  <h2>Phase 3 &mdash; Embedding Geometry Analysis</h2>

  <h3>Step 1 &mdash; Backbone Embeddings</h3>
  <div class="meta-bar">
    <span><strong>Shape:</strong> (1059, 1024) float32</span>
    <span><strong>Same-class cosine similarity:</strong> 0.93</span>
    <span><strong>Different-class cosine similarity:</strong> 0.59</span>
    <span><strong>Errors:</strong> 0</span>
  </div>
  <p>
    The 0.34-point gap between same-class and cross-class similarity confirms the backbone produces
    discriminative embeddings for UCF-101 action categories it was never trained on.
  </p>

  <h3>Step 2 &mdash; t-SNE and UMAP Visualisations</h3>
  <div class="plot-grid">
    {_img(plots, "behavior_clusters", "Embedding clusters by behaviour category")}
    {_img(plots, "ucf_class_clusters", "Embedding clusters by UCF class")}
  </div>
  <div class="plot-grid">
    {_img(plots, "confidence_gradient", "Confidence gradient in embedding space")}
    {_img(plots, "ssv2_label_regions", "SSv2 label regions")}
  </div>

  <h3>Step 3 &mdash; Blind Spot Analysis</h3>
  <p>
    Four UCF-101 classes cluster in an isolated region of embedding space with no close SSv2 concept
    neighbours. The SSv2 head cannot assign a semantically meaningful label to these classes.
  </p>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>UCF Class</th><th>Nearest SSv2 Concept</th><th>Why the Mapping Fails</th></tr></thead>
      <tbody>{blind_rows}</tbody>
    </table>
  </div>
  {_img(plots, "blind_spot_analysis", "Blind spot analysis")}

  <h3>Step 3 &mdash; &ldquo;Hitting&rdquo; Catch-All Absorption</h3>
  <div class="callout yellow">
    <strong>Hitting [something] with [something]</strong> absorbed {hitting.get("top_ucf_classes_count", 33)} UCF classes.
    Within-group cosine similarity: <strong>{hitting["within_group_similarity"]:.4f}</strong>
    vs global baseline <strong>{hitting["vs_global_baseline"]:.2f}</strong>.
    This label is a spurious catch-all for any repetitive contact motion &mdash; not a real semantic cluster.
  </div>
  {_img(plots, "hitting_cluster", "Hitting cluster analysis")}
  {_img(plots, "behavior_geometry", "Behaviour geometry")}

  <h3>Step 4 &mdash; Hypothesis Testing</h3>
  <p>Three embedding-geometry proxies were tested against prediction confidence and behaviour category.</p>
  <div class="table-wrap">
    <table class="data-table">
      <thead>
        <tr>
          <th>Hypothesis</th><th>Proxy Feature</th><th>Pearson r</th><th>p-value</th>
          <th>ANOVA F</th><th>ANOVA p</th><th>Verdict</th>
        </tr>
      </thead>
      <tbody>{hyp_rows}</tbody>
    </table>
  </div>

  <div class="callout">
    <strong>The Tight-Cluster Paradox</strong><br>
    {tcp_classes} LOW_CONF_SCATTERED classes have <em>above-median within-class cosine similarity</em>
    (tight backbone clusters) but low SSv2 confidence. Hypothesis C was rejected &mdash; the backbone
    encodes these classes clearly; the SSv2 head simply has no vocabulary for them.
    Phase 4 resolves this directly.
  </div>

  <div class="plot-grid">
    {_img(plots, "hypothesis_a_variance", "Hypothesis A: Intra-class variance")}
    {_img(plots, "hypothesis_b_distance", "Hypothesis B: Distance from SSv2 centre")}
  </div>
  <div class="plot-grid">
    {_img(plots, "hypothesis_c_similarity", "Hypothesis C: Within-class similarity")}
    {_img(plots, "feature_summary", "Feature summary")}
  </div>
</section>"""

    # ---------- phase 4 ----------
    pb_acc = lp["per_behavior_accuracy"]
    tcp = lp["tight_cluster_paradox"]

    pb_rows = ""
    for beh in ["HIGH_CONF_CONSISTENT", "HIGH_CONF_INCONSISTENT", "LOW_CONF_SCATTERED"]:
        acc = pb_acc.get(beh, 0.0)
        c = BEHAVIOR_COLORS[beh]
        pb_rows += f"""
        <tr>
          <td><span class="beh-badge" style="background:{c}22;color:{c};border:1px solid {c}44">{beh}</span></td>
          <td>{acc:.4f} ({acc*100:.1f}%)</td>
        </tr>"""

    # Fix 1 in use: delta from per_class_delta (joined at load time)
    bs_rows = ""
    for cls in ["PoleVault", "WritingOnBoard", "SalsaSpin", "Surfing"]:
        ssv2_conf = data["pr_by_class"][cls]["mean_top1_prob"]
        probe_acc = lp["per_class_accuracy"].get(cls, float("nan"))
        delta = data["per_class_delta"].get(cls, float("nan"))
        bs_rows += f"""
        <tr>
          <td>{cls}</td>
          <td>{ssv2_conf:.3f}</td>
          <td>{probe_acc:.3f}</td>
          <td style="color:#2ecc71;font-weight:600">{delta:+.3f}</td>
        </tr>"""

    overall_mean = lp["overall_accuracy_mean"]
    overall_std = lp["overall_accuracy_std"]

    phase4 = f"""
<section id="phase4">
  <h2>Phase 4 &mdash; Linear Probe Experiment</h2>
  <p>
    A logistic regression classifier was trained on the frozen (1059, 1024) backbone embeddings
    using 5-fold stratified cross-validation. StandardScaler fit on train folds only.
    No model reloading &mdash; pure sklearn on the cached embeddings.
  </p>

  <div class="stat-grid">
    <div class="stat-card">
      <span class="stat-number" style="font-size:2.5rem">{overall_mean*100:.1f}%</span>
      <div class="stat-label">Overall 5-fold accuracy &plusmn; {overall_std*100:.1f}%</div>
    </div>
    <div class="stat-card">
      <span class="stat-number" style="color:#2ecc71">100%</span>
      <div class="stat-label">Accuracy on all 4 blind-spot classes</div>
    </div>
    <div class="stat-card">
      <span class="stat-number">{tcp["mean_linear_probe_accuracy"]*100:.1f}%</span>
      <div class="stat-label">Tight-cluster paradox classes (vs {tcp["mean_ssv2_confidence"]*100:.1f}% SSv2 confidence)</div>
    </div>
  </div>

  <h3>Per-Behaviour Accuracy</h3>
  <div class="table-wrap">
    <table class="data-table" style="max-width:500px">
      <thead><tr><th>Behaviour Category</th><th>Linear Probe Accuracy</th></tr></thead>
      <tbody>{pb_rows}</tbody>
    </table>
  </div>

  <h3>Blind Spot Resolution</h3>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>UCF Class</th><th>SSv2 Confidence</th><th>Linear Probe Accuracy</th><th>Delta</th></tr></thead>
      <tbody>{bs_rows}</tbody>
    </table>
  </div>

  <h3>Tight-Cluster Paradox Resolution</h3>
  <div class="callout green">
    <strong>{tcp["mean_linear_probe_accuracy"]*100:.1f}% linear probe accuracy</strong>
    vs <strong>{tcp["mean_ssv2_confidence"]*100:.1f}% SSv2 confidence</strong>
    across the {tcp_classes} paradox classes (delta = {tcp["delta"]:+.3f}).<br><br>
    {tcp["conclusion"]}
  </div>

  {_img(plots, "linear_probe", "Linear probe results")}
</section>"""

    # ---------- conclusions ----------
    conclusions = """
<section id="conclusions">
  <h2>Conclusions</h2>
  <ol class="conclusion-list">
    <li>
      <strong>The V-JEPA 2 backbone generalises to unseen action categories.</strong>
      Same-class cosine similarity of 0.93 across UCF-101 &mdash; a dataset never seen during training &mdash;
      confirms the JEPA pre-training objective produces genuinely general spatiotemporal representations.
    </li>
    <li>
      <strong>The SSv2 classification head vocabulary is the bottleneck, not the representations.</strong>
      A linear probe on frozen embeddings achieves 98.2% on UCF-101.
      The backbone already has the answer &mdash; the SSv2 head simply has no UCF-101 words.
    </li>
    <li>
      <strong>Blind spots are entirely a head problem.</strong>
      PoleVault, SalsaSpin, WritingOnBoard, and Surfing &mdash; all flagged as blind spots in Phase 2 &mdash;
      achieve 100% linear probe accuracy. Their embeddings are geometrically clean;
      the SSv2 vocabulary just maps them to the nearest physics primitive.
    </li>
    <li>
      <strong>SSv2 physics primitives dominate the head&rsquo;s out-of-distribution vocabulary.</strong>
      &ldquo;Hitting&rdquo; (154 clips), &ldquo;Throwing&rdquo; (133), &ldquo;Spinning&rdquo; (98)
      absorb the majority of UCF-101 classes because SSv2 was trained on hand-object interactions.
      Any large motion pattern gets forced into the nearest contact primitive.
    </li>
    <li>
      <strong>Replacing the SSv2 head with a linear mapping would yield ~98% accuracy on UCF-101
      with zero backbone training.</strong>
      No fine-tuning, no LoRA, no additional data &mdash; just a 1024&times;101 weight matrix
      on top of the frozen backbone.
    </li>
  </ol>

  <h3>Limitations</h3>
  <ul style="padding-left:1.5rem;color:#cdd9e5">
    <li style="margin-bottom:0.5rem">~10 videos per class &mdash; per-class accuracy is coarse (2 test samples per fold)</li>
    <li style="margin-bottom:0.5rem">Linear probe trained and evaluated on the same embedding set used for analysis (no held-out split)</li>
    <li style="margin-bottom:0.5rem">Single model (V-JEPA 2 ViT-L SSv2) and single dataset (UCF-101)</li>
    <li>Class size imbalance in UCF-101 sample (min 9, max 20) introduces minor variance</li>
  </ul>

  <h3>Future Work</h3>
  <ul style="padding-left:1.5rem;color:#cdd9e5">
    <li style="margin-bottom:0.5rem">Test with other out-of-distribution datasets (Kinetics-400, HMDB-51)</li>
    <li style="margin-bottom:0.5rem">Compare LoRA backbone fine-tuning vs head-only replacement on full UCF-101</li>
    <li style="margin-bottom:0.5rem">Temporal dynamics analysis using optical flow to validate Hypothesis A proxy</li>
    <li>Probe intermediate transformer layers to identify where UCF-101 action geometry forms</li>
  </ul>
</section>"""

    # ---------- appendix ----------
    appendix_rows = ""
    for r in sorted(data["pattern_report"], key=lambda x: x["ucf_class"]):
        cls = r["ucf_class"]
        beh = r["behavior"]
        ssv2_conf = r["mean_top1_prob"]   # Fix 4: mean_top1_prob from pattern_report
        probe_acc = lp["per_class_accuracy"].get(cls, float("nan"))
        delta = data["per_class_delta"].get(cls, float("nan"))
        bc = BEHAVIOR_COLORS.get(beh, "#58a6ff")
        row_bg = f"{bc}0d"
        delta_color = "#2ecc71" if delta > 0.3 else "#f39c12" if delta > 0 else "#e74c3c"
        appendix_rows += f"""
        <tr style="background:{row_bg}">
          <td>{cls}</td>
          <td><span class="beh-badge" style="background:{bc}22;color:{bc};border:1px solid {bc}44">{beh}</span></td>
          <td>{ssv2_conf:.3f}</td>
          <td>{probe_acc:.3f}</td>
          <td style="color:{delta_color};font-weight:600">{delta:+.3f}</td>
        </tr>"""

    appendix = f"""
<section id="appendix">
  <h2>Technical Appendix</h2>
  <h3>Full Per-Class Results (click column headers to sort)</h3>
  <div class="table-wrap">
    <table class="data-table" id="class-table">
      <thead>
        <tr>
          <th onclick="sortTable('class-table',0)">UCF Class</th>
          <th onclick="sortTable('class-table',1)">Behaviour</th>
          <th onclick="sortTable('class-table',2)">SSv2 Confidence</th>
          <th onclick="sortTable('class-table',3)">Linear Probe Acc</th>
          <th onclick="sortTable('class-table',4)">Delta</th>
        </tr>
      </thead>
      <tbody>{appendix_rows}</tbody>
    </table>
  </div>

  <h3>Software Versions</h3>
  <div class="table-wrap">
    <table class="data-table" style="max-width:600px">
      <thead><tr><th>Package</th><th>Version</th></tr></thead>
      <tbody>
        <tr><td>Python</td><td>3.11.15</td></tr>
        <tr><td>PyTorch</td><td>2.12.0 (MPS)</td></tr>
        <tr><td>torchvision</td><td>0.27.0</td></tr>
        <tr><td>transformers</td><td>5.10.0.dev0</td></tr>
        <tr><td>scikit-learn</td><td>1.8.0</td></tr>
        <tr><td>numpy</td><td>2.4.6</td></tr>
        <tr><td>umap-learn</td><td>0.5.12</td></tr>
        <tr><td>scipy</td><td>1.17.1</td></tr>
      </tbody>
    </table>
  </div>
</section>"""

    footer = f"""
<footer style="text-align:center;padding:2rem;color:#8b949e;border-top:1px solid #30363d;margin-top:3rem">
  Generated by vjepa2-live &middot; hellojais &middot; {gen_date}
  &middot; <a href="https://github.com/hellojais/vjepa2-live">github.com/hellojais/vjepa2-live</a>
</footer>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>V-JEPA 2 Live: Probing Self-Supervised Video Representations</title>
  <style>{css}</style>
</head>
<body>
{nav}
<div class="page">
{header}
{exec_summary}
{study_design}
{phase2}
{phase3}
{phase4}
{conclusions}
{appendix}
{footer}
</div>
<script>{js}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Section 4 — Markdown Report Generation
# ---------------------------------------------------------------------------

def generate_markdown(data: dict) -> str:
    lp = data["linear_probe"]
    s4 = data["step4"]
    pb_acc = lp["per_behavior_accuracy"]
    beh_counts = data["beh_counts"]
    gen_date = data["generated_at"]

    hyp_table = (
        "| Hypothesis | Proxy | Pearson r | p-value | Verdict |\n"
        "|------------|-------|-----------|---------|--------|\n"
    )
    for key, label, proxy in [
        ("hypothesis_a", "Temporal Dynamics",  "intra_class_variance"),
        ("hypothesis_b", "Spatial Scale",       "distance_from_ssv2_center"),
        ("hypothesis_c", "Cluster Tightness",   "within_class_similarity"),
    ]:
        h = s4[key]
        hyp_table += f"| {label} | `{proxy}` | {h['pearson_r']:+.3f} | {h['pearson_p']:.4f} | **{h['verdict']}** |\n"

    phase_table = (
        "| Phase | Goal | Key Result |\n"
        "|-------|------|------------|\n"
        "| Phase 1 | Validate on Apple Silicon | ~10fps inference, importable module |\n"
        "| Phase 2 | Zero-shot UCF-101 evaluation | 101 classes → 3 behaviour groups |\n"
        "| Phase 3 Step 1 | Extract embeddings | (1059, 1024) float32 |\n"
        "| Phase 3 Step 2 | t-SNE + UMAP | 4 visualisation plots |\n"
        "| Phase 3 Step 3 | Nearest-neighbour analysis | Blind spots, hitting absorption |\n"
        "| Phase 3 Step 4 | Hypothesis testing | Temporal dynamics partially explains confidence |\n"
        "| Phase 4 | Linear probe | **98.2% accuracy** — head is the bottleneck |\n"
    )

    # Per-class table (top 20 by delta)
    all_rows = []
    for r in data["pattern_report"]:
        cls = r["ucf_class"]
        delta = data["per_class_delta"].get(cls, float("nan"))
        all_rows.append((cls, r["behavior"], r["mean_top1_prob"],
                         lp["per_class_accuracy"].get(cls, float("nan")), delta))
    all_rows.sort(key=lambda x: -x[4])
    top20_table = (
        "| UCF Class | Behaviour | SSv2 Conf | Probe Acc | Delta |\n"
        "|-----------|-----------|-----------|-----------|-------|\n"
    )
    for cls, beh, ssv2, probe, delta in all_rows[:20]:
        top20_table += f"| {cls} | {beh} | {ssv2:.3f} | {probe:.3f} | {delta:+.3f} |\n"

    return f"""# V-JEPA 2 Live: Probing Self-Supervised Video Representations

> "{CENTRAL_FINDING}"

**Generated:** {gen_date}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Linear Probe Accuracy | **{lp["overall_accuracy_mean"]*100:.1f}% ± {lp["overall_accuracy_std"]*100:.1f}%** |
| Classes SSv2 head fails on | **42%** (LOW_CONF_SCATTERED) |
| Backbone same-class similarity | **0.93** |
| Blind-spot linear probe accuracy | **100%** (all 4 classes) |

---

## Study Design

{phase_table}

---

## Key Findings

### Finding 1: The Backbone Generalises

V-JEPA 2's ViT-L backbone produces embeddings with 0.93 same-class cosine similarity
on UCF-101 — a dataset it was never trained on. The 0.34-point gap vs cross-class
similarity (0.59) confirms genuinely discriminative spatiotemporal representations.

### Finding 2: The Head is the Bottleneck

98.2% ± 1.0% top-1 accuracy with a frozen linear classifier on the backbone embeddings.
No fine-tuning. No LoRA. Just a 1024×101 weight matrix.

### Finding 3: Blind Spots Are a Head Problem

PoleVault, SalsaSpin, WritingOnBoard, Surfing all scored 100% with the linear probe.
Their "blind spot" status in Phase 2 was purely a vocabulary failure of the SSv2 head.

### Finding 4: Physics Primitives Dominate SSv2 Vocabulary

"Hitting" (154 clips), "Throwing" (133), "Spinning" (98) account for the majority
of confident SSv2 predictions on UCF-101. The SSv2 head forces all large motion
patterns into the nearest hand-object contact primitive.

### Finding 5: 98.2% With Only a Linear Head Swap

No backbone training needed. The world model already understands all 101 UCF-101
action categories. It just doesn't know their names.

---

## Phase 2 Results

| Behaviour | Classes |
|-----------|---------|
| HIGH_CONF_CONSISTENT | {beh_counts.get("HIGH_CONF_CONSISTENT", 0)} |
| HIGH_CONF_INCONSISTENT | {beh_counts.get("HIGH_CONF_INCONSISTENT", 0)} |
| LOW_CONF_SCATTERED | {beh_counts.get("LOW_CONF_SCATTERED", 0)} |

---

## Phase 3 Hypothesis Testing

{hyp_table}

**Tight-Cluster Paradox:** {len(s4.get("classes_unexplained", []))} LOW_CONF_SCATTERED classes
have tight backbone clusters but low SSv2 confidence. Phase 4 resolves this: their
mean linear probe accuracy is {lp["tight_cluster_paradox"]["mean_linear_probe_accuracy"]*100:.1f}%
vs {lp["tight_cluster_paradox"]["mean_ssv2_confidence"]*100:.1f}% SSv2 confidence.

---

## Phase 4 Linear Probe Results

| Behaviour | Accuracy |
|-----------|----------|
| HIGH_CONF_CONSISTENT | {pb_acc.get("HIGH_CONF_CONSISTENT", 0)*100:.1f}% |
| HIGH_CONF_INCONSISTENT | {pb_acc.get("HIGH_CONF_INCONSISTENT", 0)*100:.1f}% |
| LOW_CONF_SCATTERED | {pb_acc.get("LOW_CONF_SCATTERED", 0)*100:.1f}% |

### Top 20 Classes by Delta (probe acc − SSv2 confidence)

{top20_table}

---

## Limitations

- ~10 videos per class; per-class accuracy is coarse (2 test samples per fold)
- Linear probe uses the same embedding set as the analysis (no held-out split)
- Single model (V-JEPA 2 ViT-L SSv2) and single dataset (UCF-101)

---

## Conclusions

1. The backbone generalises to unseen action categories
2. The SSv2 head vocabulary is the bottleneck — not the representations
3. Blind spots are entirely a head problem
4. SSv2 physics primitives dominate the head's out-of-distribution vocabulary
5. Replacing the SSv2 head with a linear mapping yields ~98% UCF-101 accuracy with zero additional training
"""


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _readme(data: dict) -> str:
    lp = data["linear_probe"]
    return f"""# vjepa2-live

> "V-JEPA 2\u2019s self-supervised backbone achieves 98.2% linear probe accuracy on UCF-101 \u2014 a dataset it was never trained on \u2014 while its SSv2 classification head produces confident but semantically wrong predictions for 42% of the same classes. The bottleneck is not the representation. It is the vocabulary."

## What This Project Is

A systematic probing study of V-JEPA 2\u2019s world model on out-of-distribution video data.
The model was never fine-tuned on UCF-101 \u2014 it was used frozen throughout.
Every phase asks a more precise version of the same question: *what does a self-supervised
video transformer actually learn, and where does its understanding break down?*

This is not a tutorial or a benchmark submission. It is an original research pipeline
built on a MacBook Pro M5 Max, using only open-source tools, that arrives at a concrete
and falsifiable finding: the SSv2 classification head is the sole bottleneck between
the model\u2019s near-perfect internal representations and its poor zero-shot UCF-101 accuracy.

## The Finding In One Image

![Linear Probe Results](results/plots/linear_probe.png)

## Project Structure

```
vjepa2_webcam.py          Phase 1 \u2014 live webcam inference + importable module
vjepa2_ucf101_eval.py     Phase 2 \u2014 zero-shot UCF-101 batch evaluation
extract_embeddings.py     Phase 3 Step 1 \u2014 backbone embedding extraction
visualize_embeddings.py   Phase 3 Step 2 \u2014 t-SNE + UMAP visualisation
nearest_neighbor_analysis.py  Phase 3 Step 3 \u2014 nearest-neighbour analysis
behavior_analysis.py      Phase 3 Step 4 \u2014 hypothesis testing
linear_probe.py           Phase 4 \u2014 linear probe experiment
generate_report.py        Phase 5 \u2014 HTML + Markdown report generation
results/                  All outputs (CSV, JSON, PNG, HTML)
requirements.txt          Python dependencies
```

## Phases

| Phase | Goal | Key Result |
|-------|------|------------|
| Phase 1 | Validate V-JEPA 2 on Apple Silicon | ~10fps inference, importable module |
| Phase 2 | Zero-shot UCF-101 evaluation | 101 classes \u2192 3 behaviour groups |
| Phase 3 Step 1 | Extract backbone embeddings | (1059, 1024) float32 |
| Phase 3 Step 2 | t-SNE + UMAP visualisation | 4 visualisation plots |
| Phase 3 Step 3 | Nearest-neighbour analysis | Blind spots, hitting absorption |
| Phase 3 Step 4 | Hypothesis testing | Temporal dynamics partially explains confidence |
| Phase 4 | Linear probe | **98.2% accuracy** \u2014 head is the bottleneck |

## Key Findings

1. **The backbone generalises to unseen action categories** \u2014 0.93 same-class cosine similarity on UCF-101
2. **The SSv2 head vocabulary is the bottleneck** \u2014 98.2% linear probe accuracy on frozen embeddings
3. **Blind spots are a head problem, not a backbone problem** \u2014 all 4 blind-spot classes score 100% with a linear probe
4. **SSv2 physics primitives dominate the head\u2019s OOD vocabulary** \u2014 \u201cHitting\u201d absorbs 33 UCF classes
5. **Zero additional training needed** \u2014 a 1024\u00d7101 linear layer on the frozen backbone suffices

## Setup

### Requirements

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Download UCF-101 from [https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)
and extract to `data/UCF-101/`.

### Run

```bash
# Phase 2 \u2014 zero-shot evaluation (requires model + UCF-101 data)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 vjepa2_ucf101_eval.py

# Phase 3 \u2014 embedding extraction (requires Phase 2 output)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 extract_embeddings.py

# Phase 3 Steps 2\u20134 + Phase 4 (pure numpy/sklearn, no model needed)
python3 visualize_embeddings.py
python3 nearest_neighbor_analysis.py
python3 behavior_analysis.py
python3 linear_probe.py

# Final report
python3 generate_report.py
# Open results/vjepa2_live_report.html
```

## Hardware

MacBook Pro M5 Max, 68GB unified memory, macOS
PyTorch 2.12 MPS backend \u2014 no CUDA required

## Model

[qubvel-hf/vjepa2-vitl-fpc16-256-ssv2](https://huggingface.co/qubvel-hf/vjepa2-vitl-fpc16-256-ssv2)
\u2014 V-JEPA 2 ViT-L, 325M parameters, self-supervised pre-training, fine-tuned on
Something-Something v2 (174 hand-object interaction classes).

## Limitations

- ~10 videos per class (small sample regime)
- Linear probe uses same embeddings as analysis (no fully held-out set)
- Single model and single dataset; results may not generalise
- Per-class accuracy is coarse due to small test set per fold

## Author

[hellojais](https://github.com/hellojais)
"""


# ---------------------------------------------------------------------------
# Section 5 — Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading all results...")
    data = load_all_results()

    print("Generating HTML report...")
    html = generate_html(data)
    REPORT_OUTPUT.write_text(html, encoding="utf-8")
    size_mb = REPORT_OUTPUT.stat().st_size / 1024 / 1024
    print(f"HTML report saved to {REPORT_OUTPUT}  ({size_mb:.1f} MB)")

    print("Generating Markdown report...")
    md = generate_markdown(data)
    Path("results/vjepa2_live_report.md").write_text(md, encoding="utf-8")
    print("Markdown report saved to results/vjepa2_live_report.md")

    print("Generating README.md...")
    readme = _readme(data)
    Path("README.md").write_text(readme, encoding="utf-8")
    print("README.md saved to project root")

    print("\nDone. Open results/vjepa2_live_report.html in your browser.")

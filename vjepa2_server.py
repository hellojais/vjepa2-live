"""
vjepa2_server.py — Web-based live inference server for V-JEPA 2.

Imports model, processor, device, and run_inference directly from
vjepa2_webcam (module-level) so weights are never loaded twice.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 vjepa2_server.py

Then open:  http://127.0.0.1:7860
Allow camera access in the browser when prompted.
"""

import base64
import collections
import threading
import webbrowser

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from vjepa2_webcam import device, model, processor, run_inference

# ---------------------------------------------------------------------------
# Server-side frame buffer  (single-user local demo — no session management)
# ---------------------------------------------------------------------------
NUM_FRAMES = 16
STRIDE = 8

_buffer: collections.deque = collections.deque(maxlen=NUM_FRAMES)
_frames_since_inference: int = 0
_latest_predictions: list = []

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML / CSS / JS — single self-contained page
# ---------------------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>V-JEPA 2 · Live Inference</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #07070f;
      color: #d4d4d8;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                   "Segoe UI", system-ui, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* ── Header ─────────────────────────────────────────────────────────── */
    header {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 18px 32px;
      border-bottom: 1px solid #18181f;
    }

    .logo {
      width: 34px;
      height: 34px;
      border-radius: 9px;
      background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      flex-shrink: 0;
    }

    header h1 {
      font-size: 17px;
      font-weight: 600;
      letter-spacing: -0.4px;
      color: #e4e4e7;
    }

    header h1 span {
      color: #6366f1;
    }

    .header-pill {
      margin-left: auto;
      font-size: 11px;
      font-weight: 500;
      padding: 4px 10px;
      border-radius: 20px;
      background: #12121a;
      border: 1px solid #22222e;
      color: #71717a;
      letter-spacing: 0.3px;
    }

    /* ── Main layout ─────────────────────────────────────────────────────── */
    main {
      flex: 1;
      display: flex;
      gap: 24px;
      padding: 28px 32px;
      max-width: 1180px;
      margin: 0 auto;
      width: 100%;
    }

    /* ── Video panel ─────────────────────────────────────────────────────── */
    .video-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-width: 0;
    }

    .video-wrapper {
      position: relative;
      border-radius: 14px;
      overflow: hidden;
      background: #0d0d14;
      border: 1px solid #18181f;
      aspect-ratio: 4 / 3;
    }

    #video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scaleX(-1);
      display: block;
    }

    /* corner accent */
    .video-wrapper::before {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      background: linear-gradient(
        160deg,
        rgba(99, 102, 241, 0.07) 0%,
        transparent 50%
      );
      pointer-events: none;
      z-index: 1;
    }

    #canvas { display: none; }

    .status-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: #52525b;
    }

    .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      flex-shrink: 0;
      background: #22c55e;
    }

    .dot.idle    { background: #f59e0b; }
    .dot.live    { background: #22c55e; animation: blink 2s ease-in-out infinite; }
    .dot.buffer  { background: #6366f1; animation: blink 1s ease-in-out infinite; }
    .dot.error   { background: #ef4444; }

    @keyframes blink {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.25; }
    }

    /* ── Results panel ───────────────────────────────────────────────────── */
    .results-panel {
      width: 310px;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .section-label {
      font-size: 10.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: #3f3f5a;
    }

    /* ── Prediction cards ────────────────────────────────────────────────── */
    .predictions {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .card {
      background: #0d0d14;
      border: 1px solid #18181f;
      border-radius: 12px;
      padding: 14px 16px;
      transition: border-color 0.3s, background 0.3s;
    }

    .card.top {
      border-color: rgba(99, 102, 241, 0.5);
      background: #10101a;
    }

    .card .row {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      margin-bottom: 10px;
    }

    .rank {
      width: 20px;
      height: 20px;
      border-radius: 6px;
      background: #18181f;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      font-weight: 700;
      color: #52525b;
      flex-shrink: 0;
    }

    .top .rank {
      background: #6366f1;
      color: #fff;
    }

    .label {
      flex: 1;
      font-size: 13px;
      font-weight: 500;
      color: #c4c4c8;
      word-break: break-word;
      line-height: 1.4;
    }

    .top .label { color: #a5b4fc; }

    .pct {
      font-size: 13px;
      font-weight: 700;
      color: #6366f1;
      flex-shrink: 0;
    }

    .bar-track {
      height: 3px;
      background: #18181f;
      border-radius: 2px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: 2px;
      background: linear-gradient(90deg, #6366f1, #a855f7);
      transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Waiting placeholder ─────────────────────────────────────────────── */
    .placeholder {
      background: #0d0d14;
      border: 1px dashed #1e1e2e;
      border-radius: 12px;
      padding: 36px 20px;
      text-align: center;
      color: #3f3f5a;
      font-size: 13px;
      line-height: 1.9;
    }

    .placeholder .icon { font-size: 30px; margin-bottom: 10px; }

    /* ── Info box ────────────────────────────────────────────────────────── */
    .info-box {
      background: #0d0d14;
      border: 1px solid #18181f;
      border-radius: 12px;
      padding: 14px 16px;
      font-size: 12px;
      color: #3f3f5a;
      line-height: 1.85;
    }

    .info-box b { color: #52525b; }

    /* ── Footer ──────────────────────────────────────────────────────────── */
    footer {
      padding: 14px 32px;
      border-top: 1px solid #18181f;
      font-size: 11px;
      color: #27272a;
      text-align: center;
      letter-spacing: 0.2px;
    }
  </style>
</head>
<body>

  <header>
    <div class="logo">&#x1F9E0;</div>
    <h1>V-JEPA <span>2</span> &mdash; Live Inference</h1>
    <span class="header-pill">vitl-fpc16-256-ssv2 &nbsp;&middot;&nbsp; MPS</span>
  </header>

  <main>
    <!-- Video feed -->
    <div class="video-panel">
      <div class="video-wrapper">
        <video id="video" autoplay playsinline muted></video>
      </div>
      <canvas id="canvas"></canvas>
      <div class="status-row">
        <div class="dot idle" id="dot"></div>
        <span id="status-text">Waiting for webcam&hellip;</span>
      </div>
    </div>

    <!-- Results -->
    <div class="results-panel">
      <div class="section-label">Top Predictions</div>

      <div class="predictions" id="predictions">
        <div class="placeholder">
          <div class="icon">&#x1F4F9;</div>
          Allow camera access to begin.<br>
          Inference starts once<br>16 frames are buffered.
        </div>
      </div>

      <div class="info-box">
        <b>Model</b><br>
        qubvel-hf/vjepa2-vitl-fpc16-256-ssv2<br><br>
        <b>Buffer</b>&nbsp; 16 frames &nbsp;&nbsp;
        <b>Stride</b>&nbsp; 8 frames<br>
        <b>Training set</b>&nbsp; Something-Something v2
      </div>
    </div>
  </main>

  <footer>
    Running locally on Apple Silicon MPS &nbsp;&middot;&nbsp; http://127.0.0.1:7860
  </footer>

  <script>
    const video      = document.getElementById('video');
    const canvas     = document.getElementById('canvas');
    const ctx        = canvas.getContext('2d');
    const predsEl    = document.getElementById('predictions');
    const dot        = document.getElementById('dot');
    const statusText = document.getElementById('status-text');

    let buffered = 0;
    const BUFFER_TARGET = 16;
    const SEND_INTERVAL_MS = 100; // ~10 fps to server

    // ── Start webcam ──────────────────────────────────────────────────────
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
      .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          canvas.width  = video.videoWidth;
          canvas.height = video.videoHeight;
          setStatus('buffer', 'Buffering frames\u2026 0 / 16');
          setInterval(captureAndSend, SEND_INTERVAL_MS);
        };
      })
      .catch(err => {
        setStatus('error', 'Camera denied: ' + err.message);
      });

    // ── Capture + send ────────────────────────────────────────────────────
    async function captureAndSend() {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const b64 = canvas.toDataURL('image/jpeg', 0.72).split(',')[1];

      buffered = Math.min(buffered + 1, BUFFER_TARGET);
      if (buffered < BUFFER_TARGET) {
        setStatus('buffer', `Buffering frames\u2026 ${buffered} / ${BUFFER_TARGET}`);
        // still send so server fills its buffer too
      }

      try {
        const res  = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame: b64 })
        });
        const data = await res.json();

        if (buffered >= BUFFER_TARGET) {
          setStatus('live', 'Live \u2014 ' + new Date().toLocaleTimeString());
          renderPredictions(data.predictions);
        }
      } catch (e) {
        setStatus('error', 'Server error: ' + e.message);
      }
    }

    // ── Render prediction cards ───────────────────────────────────────────
    function renderPredictions(preds) {
      if (!preds || preds.length === 0) return;
      predsEl.innerHTML = preds.map(([label, prob], i) => {
        const pct     = (prob * 100).toFixed(1);
        const isTop   = i === 0 ? 'top' : '';
        return `
          <div class="card ${isTop}">
            <div class="row">
              <div class="rank">${i + 1}</div>
              <div class="label">${label}</div>
              <div class="pct">${pct}%</div>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct}%"></div>
            </div>
          </div>`;
      }).join('');
    }

    // ── Status helper ─────────────────────────────────────────────────────
    function setStatus(state, text) {
      dot.className = 'dot ' + state;
      statusText.textContent = text;
    }
  </script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    global _frames_since_inference, _latest_predictions

    payload = request.get_json(force=True)
    frame_b64: str = payload.get("frame", "")

    # Decode base64 JPEG → numpy uint8 RGB
    frame_bytes = base64.b64decode(frame_b64)
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        return jsonify({"predictions": _latest_predictions})

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    _buffer.append(rgb)
    _frames_since_inference += 1

    if len(_buffer) == NUM_FRAMES and _frames_since_inference >= STRIDE:
        _latest_predictions = run_inference(_buffer, processor, model, device)
        _frames_since_inference = 0

    return jsonify({"predictions": _latest_predictions})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    url = "http://127.0.0.1:7860"
    print(f"Starting server → {url}")
    # Open browser after a short delay so Flask is ready
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=7860, debug=False)

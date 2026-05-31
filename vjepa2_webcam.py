# =============================================================================
# Section 1 — Imports
# =============================================================================
import sys
import time
import collections
import numpy as np
import cv2
import torch
from transformers import AutoVideoProcessor, AutoModelForVideoClassification

# =============================================================================
# Section 2 — Model Loading
# =============================================================================
MODEL_ID = "qubvel-hf/vjepa2-vitl-fpc16-256-ssv2"

processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)

device = torch.device("mps")
model = model.to(device)
model.eval()

print(f"Model loaded: {MODEL_ID}")
print(f"Model is on device: {next(model.parameters()).device}")

# =============================================================================
# Section 5 — Inference Function
# =============================================================================
def run_inference(frame_buffer, processor, model, device) -> list[tuple[str, float]]:
    """
    Run V-JEPA 2 inference on a buffer of frames.

    Args:
        frame_buffer: iterable of NUM_FRAMES numpy arrays, each (H, W, 3) uint8 RGB
        processor:    AutoVideoProcessor instance
        model:        AutoModelForVideoClassification instance
        device:       torch.device pointing to MPS

    Returns:
        Top-3 list of (label_string, probability_float) tuples, sorted by probability.
    """
    # Stack frames into a single (16, H, W, 3) uint8 numpy array
    clip_array = np.stack(list(frame_buffer), axis=0).astype(np.uint8)

    # Build model inputs; processor returns PyTorch tensors on CPU
    inputs = processor(videos=[clip_array], return_tensors="pt")

    # Move every input tensor to the MPS device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits               # shape: (1, num_classes)
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # shape: (num_classes,)

    top3 = torch.topk(probs, k=3)
    results = [
        (model.config.id2label[idx.item()], float(probs[idx].item()))
        for idx in top3.indices
    ]
    return results


# =============================================================================
# Main entry point — webcam loop only runs when executed directly
# =============================================================================
if __name__ == "__main__":

    # =========================================================================
    # Section 3 — Webcam Setup
    # =========================================================================
    NUM_FRAMES = 16
    STRIDE = 8

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Could not open webcam at index 0."

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened: {native_width}x{native_height} @ {native_fps:.1f} FPS")

    buffer = collections.deque(maxlen=NUM_FRAMES)
    frames_since_last_inference = 0

    # =========================================================================
    # Section 6 — Display Loop  (Section 4 frame-buffer logic lives inside)
    # =========================================================================
    latest_predictions: list[tuple[str, float]] = []

    while True:

        # ---------------------------------------------------------------------
        # Section 4 — Frame Buffer Logic
        # ---------------------------------------------------------------------
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Exiting.")
            break

        # BGR → RGB; keep as uint8 HxWx3 numpy array for the deque
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(rgb_frame)
        frames_since_last_inference += 1

        # Trigger inference when buffer is full and stride interval has passed
        if len(buffer) == NUM_FRAMES and frames_since_last_inference >= STRIDE:
            latest_predictions = run_inference(buffer, processor, model, device)
            frames_since_last_inference = 0

        # ---------------------------------------------------------------------
        # Overlay: draw predictions on the BGR display frame
        # ---------------------------------------------------------------------
        y = 30
        for rank, (label, prob) in enumerate(latest_predictions, start=1):
            text = f"{rank}. {label}: {prob:.1%}"
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 30

        cv2.imshow("V-JEPA 2 Webcam Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # =========================================================================
    # Section 7 — Clean Shutdown
    # =========================================================================
    cap.release()
    cv2.destroyAllWindows()
    print("Shutting down.")

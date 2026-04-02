# =============================================================================
# src/detection/detector.py
# Smart License Plate Detection System
# YOLOv8-based license plate detector — modular and pipeline-ready.
# =============================================================================

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

from src.detection.preprocessor import preprocess_frame


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL_PATH   = "models/detector/yolov8_license_plate.pt"
CONFIDENCE_THRESHOLD = 0.25   # Detections below this are discarded
IOU_THRESHOLD        = 0.45   # Non-Maximum Suppression overlap threshold
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# LicensePlateDetector CLASS
# =============================================================================

class LicensePlateDetector:
    """
    Detects license plates in images and video frames using YOLOv8.

    Usage:
        detector   = LicensePlateDetector()
        detections = detector.detect_frame(frame)
        crops      = detector.crop_plates(frame, detections)
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Load the YOLOv8 model. Falls back to yolov8n.pt if custom weights
        are not yet available (useful for early-stage testing).
        """
        if not Path(model_path).exists():
            print(f"[WARNING] Weights not found at '{model_path}'. Using yolov8n.pt.")
            model_path = "yolov8n.pt"   # Auto-downloads from Ultralytics

        print(f"[INFO] Loading model: {model_path}  |  Device: {DEVICE.upper()}")
        self.model  = YOLO(model_path)
        self.device = DEVICE


    # -------------------------------------------------------------------------
    # detect_frame — core method used by pipeline
    # -------------------------------------------------------------------------

    def detect_frame(self, frame, preprocess: bool = True) -> list[dict]:
        """
        Run YOLOv8 on a single frame and return structured detections.

        Args:
            frame      : BGR numpy array or image file path.
            preprocess : Run preprocessor before inference (default True).

        Returns:
            List of dicts:
                { "bbox": [x1,y1,x2,y2], "confidence": float,
                  "class_id": int, "label": str }
        """
        if preprocess and not isinstance(frame, str):
            frame = preprocess_frame(frame)

        raw = self.model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=self.device,
            verbose=False
        )

        detections = []
        for result in raw:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": round(float(box.conf[0]), 3),
                    "class_id":   int(box.cls[0]),
                    "label":      self.model.names[int(box.cls[0])]
                })

        return detections


    # -------------------------------------------------------------------------
    # detect_image — convenience wrapper for static files
    # -------------------------------------------------------------------------

    def detect_image(self, image_path: str, show: bool = False) -> list[dict]:
        """Detect plates in a saved image file and optionally display result."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        detections = self.detect_frame(frame)
        print(f"[INFO] {len(detections)} plate(s) in '{image_path}'")

        if show:
            annotated = self._draw(frame, detections)
            cv2.imshow("Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections


    # -------------------------------------------------------------------------
    # crop_plates — extract plate ROIs for OCR
    # -------------------------------------------------------------------------

    def crop_plates(self, frame, detections: list[dict]) -> list:
        """
        Crop each detected plate region from the original (un-resized) frame.

        Args:
            frame      : Original BGR frame (before preprocessing/resize).
            detections : Output of detect_frame().

        Returns:
            List of BGR crop arrays — one per detection.
        """
        crops = []
        h, w  = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Clamp coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crops.append(frame[y1:y2, x1:x2])
        return crops


    # -------------------------------------------------------------------------
    # _draw — internal helper for annotating frames
    # -------------------------------------------------------------------------

    def _draw(self, frame, detections: list[dict]):
        """Draw green bounding boxes and confidence scores on a frame copy."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(out, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    d = LicensePlateDetector()
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if src:
        d.detect_image(src, show=True)
    else:
        print("Usage: python detector.py <image_path>")
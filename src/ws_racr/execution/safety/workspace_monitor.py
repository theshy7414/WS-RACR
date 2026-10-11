#!/usr/bin/env python3
"""
workspace_monitor.py
Workspace clutter detection module.

Strategy:
  1. Crop image to the ROI (workspace region) to ignore objects outside the table.
  2. Run workpiece.pt YOLO to detect known assembly parts (fast, free, offline).
  3. Send the cropped image + YOLO summary to GPT-4o Vision.
  4. Ask VLM whether any non-assembly foreign objects are present.
  5. Return a structured ClutterCheckResult.

Standalone usage:
  python workspace_monitor.py image.jpg
  python workspace_monitor.py image.jpg --roi 0.1 0.1 0.9 0.9
  python workspace_monitor.py --camera
  python -m ws_racr.execution.safety.workspace_monitor --camera --serial <D435I_SERIAL>
  python workspace_monitor.py --list-cameras

  # D435i triple: (1) raw (2) ROI blackout (3) blackout + YOLO + English hints (VLM on by default)
  python workspace_monitor.py --camera --roi 0.05 0.1 0.95 0.9 --triple ./snap/run1
"""

from __future__ import annotations

import logging
import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np

from ws_racr.config import get_openai_client, get_yolo_weights_path

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Known assembly parts recognised by workpiece.pt — these are NOT clutter.
KNOWN_ASSEMBLY_CLASSES: frozenset[str] = frozenset({
    "bracket", "cylinder", "hex_wrench", "nut",
    "pi", "platform", "screw_big", "screw_small",
})

# Examples given to the VLM to guide its attention (not an exhaustive list).
CLUTTER_EXAMPLES = [
    "cup / bottle",
    "mouse / keyboard",
    "phone / tablet",
    "book / paper / notebook",
    "food / snack",
    "pen / stationery",
    "remote control",
    "clothing / fabric",
    "non-assembly tools (e.g. screwdriver not belonging to the kit)",
]

DEFAULT_YOLO_PATH = str(get_yolo_weights_path())


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ClutterCheckResult:
    """Structured result returned by WorkspaceClutterMonitor."""

    has_clutter: bool
    """True if foreign objects were detected."""

    severity: Literal["blocking", "warning", "safe"]
    """
    blocking: clutter clearly inside the operation zone — task should be aborted.
    warning:  clutter near the edge or a potential risk — proceed with caution.
    safe:     workspace is clean.
    """

    detected_objects: List[str]
    """Names of detected foreign objects."""

    suggestion: str
    """Short suggestion for the operator (English, ≤20 words)."""

    triggered_by: str
    """Trigger source: 'yolo_prefilter' | 'failure' | 'pre_task' | 'manual' | 'cli_test'"""

    raw_yolo_detections: List[dict] = field(default_factory=list)
    """YOLO detections of known assembly parts (for debugging / visualisation)."""

    timestamp: float = field(default_factory=time.time)

    def is_safe(self) -> bool:
        return self.severity == "safe"

    def is_blocking(self) -> bool:
        return self.severity == "blocking"

    def summary(self) -> str:
        icon = {"blocking": "[BLOCKING]", "warning": "[WARNING]", "safe": "[SAFE]"}
        tag = icon.get(self.severity, f"[{self.severity.upper()}]")
        if self.has_clutter:
            objects_str = ", ".join(self.detected_objects) if self.detected_objects else "unknown object"
            return f"{tag} Clutter: {objects_str}. {self.suggestion}"
        return f"{tag} Workspace is clean."


# =============================================================================
# Main class
# =============================================================================

class WorkspaceClutterMonitor:
    """
    Workspace clutter detector using YOLO + GPT-4o Vision.

    Usage:
        monitor = WorkspaceClutterMonitor(roi=(0.05, 0.1, 0.95, 0.9))
        result = monitor.check(image_bgr, triggered_by="pre_task")
        if result and result.is_blocking():
            print(result.suggestion)
    """

    def __init__(
        self,
        openai_client=None,
        yolo_model=None,
        yolo_model_path: str = DEFAULT_YOLO_PATH,
        vlm_model: str = "gpt-4o",
        cooldown_seconds: float = 30.0,
        max_calls_per_task: int = 5,
        jpeg_quality: int = 85,
        roi: Optional[Tuple[float, float, float, float]] = None,
        roi_is_ratio: bool = True,
        roi_polygon: Optional[List[Tuple[float, float]]] = None,
        roi_polygon_is_ratio: bool = False,
    ):
        """
        Args:
            openai_client:      Pre-built OpenAI client (auto-created from env if None).
            yolo_model:         Pre-loaded YOLO model (lazy-loaded if None).
            yolo_model_path:    Path to workpiece.pt.
            vlm_model:          Vision model name (default: gpt-4o).
            cooldown_seconds:   Minimum interval between VLM calls (seconds).
            max_calls_per_task: Maximum VLM calls per task.
            jpeg_quality:       JPEG compression quality for the encoded image (1-100).
            roi:                Region of interest as (x1, y1, x2, y2).
                                If roi_is_ratio=True (default), values are 0-1 fractions
                                of image width/height.  E.g. (0.05, 0.1, 0.95, 0.9).
                                If roi_is_ratio=False, values are absolute pixel coords.
                                None means use the full image.
            roi_is_ratio:       Whether roi values are fractions (True) or pixels (False).
            roi_polygon:        Polygon ROI points [(x1,y1), (x2,y2), ...].
                                When set, polygon ROI takes priority over rectangle ROI.
            roi_polygon_is_ratio: Whether polygon points are 0-1 fractions.
        """
        self._client = openai_client
        self._yolo = yolo_model
        self._yolo_model_path = yolo_model_path
        self.vlm_model = vlm_model
        self.cooldown_seconds = cooldown_seconds
        self.max_calls_per_task = max_calls_per_task
        self.jpeg_quality = jpeg_quality
        self.roi = roi
        self.roi_is_ratio = roi_is_ratio
        self.roi_polygon = roi_polygon
        self.roi_polygon_is_ratio = roi_polygon_is_ratio

        self._last_call_time: float = 0.0
        self._calls_this_task: int = 0
        self._last_result: Optional[ClutterCheckResult] = None
        self._last_roi_box: Optional[Tuple[int, int, int, int]] = None
        self._last_roi_polygon: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        image: np.ndarray,
        triggered_by: str = "manual",
        force: bool = False,
    ) -> Optional[ClutterCheckResult]:
        """
        Main entry point: given a BGR image, return a clutter check result.

        Args:
            image:        BGR numpy array (full camera frame).
            triggered_by: Label for logging / audit trail.
            force:        Bypass cooldown (use for post-failure emergency checks).

        Returns:
            ClutterCheckResult, or None if on cooldown / limit reached / API error.
        """
        if not force and self._is_on_cooldown():
            remaining = self.cooldown_seconds - (time.time() - self._last_call_time)
            logger.info(f"[WorkspaceMonitor] On cooldown, skipping VLM ({remaining:.0f}s remaining)")
            return self._last_result

        if not force and self._calls_this_task >= self.max_calls_per_task:
            logger.info(f"[WorkspaceMonitor] Per-task call limit reached ({self.max_calls_per_task})")
            return self._last_result

        # Crop to ROI before analysis
        roi_image, roi_box, roi_polygon = self._apply_roi(image)
        self._last_roi_box = roi_box
        self._last_roi_polygon = roi_polygon

        # Step 1: YOLO detection of known assembly parts
        yolo_detections = self._run_yolo(roi_image)
        if yolo_detections:
            names = [f"{d['class_name']}({d['confidence']:.0%})" for d in yolo_detections]
            logger.info(f"[WorkspaceMonitor] YOLO parts detected: {', '.join(names)}")
        else:
            logger.info("[WorkspaceMonitor] YOLO: no known assembly parts detected")

        # Step 2: VLM call
        result = self._call_vlm(roi_image, yolo_detections, triggered_by)
        if result is not None:
            self._last_call_time = time.time()
            self._calls_this_task += 1
            self._last_result = result

        return result

    def reset_for_new_task(self):
        """Call at the start of each new task to reset the per-task counter."""
        self._calls_this_task = 0
        self._last_result = None
        logger.info("[WorkspaceMonitor] Counter reset for new task.")

    def resolve_roi_pixels(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return the ROI as absolute pixel coords (x1, y1, x2, y2), or None."""
        if self.roi is None:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = self.roi
        if self.roi_is_ratio:
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return x1, y1, x2, y2

    def resolve_roi_polygon_pixels(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return polygon ROI points as absolute pixel coordinates Nx2, or None."""
        if not self.roi_polygon:
            return None
        h, w = image.shape[:2]
        points: List[List[int]] = []
        for px, py in self.roi_polygon:
            if self.roi_polygon_is_ratio:
                x = int(px * w)
                y = int(py * h)
            else:
                x = int(px)
                y = int(py)
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            points.append([x, y])
        if len(points) < 3:
            return None
        return np.array(points, dtype=np.int32)

    @property
    def last_result(self) -> Optional[ClutterCheckResult]:
        return self._last_result

    @property
    def last_roi_box(self) -> Optional[Tuple[int, int, int, int]]:
        return self._last_roi_box

    @property
    def last_roi_polygon(self) -> Optional[np.ndarray]:
        return self._last_roi_polygon

    def roi_mask_uint8(self, image: np.ndarray) -> np.ndarray:
        """
        Build an H×W mask: 255 inside ROI, 0 outside (uint8).
        Polygon ROI wins over axis-aligned rectangle; if neither is set, all 255.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        polygon = self.resolve_roi_polygon_pixels(image)
        if polygon is not None:
            cv2.fillPoly(mask, [polygon], 255)
            return mask
        box = self.resolve_roi_pixels(image)
        if box is not None:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 255
            return mask
        mask[:] = 255
        return mask

    def blackout_outside_roi(self, image: np.ndarray) -> np.ndarray:
        """Full-frame BGR image with pixels outside the ROI set to black."""
        mask = self.roi_mask_uint8(image)
        return cv2.bitwise_and(image, image, mask=mask)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_on_cooldown(self) -> bool:
        return (time.time() - self._last_call_time) < self.cooldown_seconds

    def _apply_roi(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        Apply ROI and return:
          (roi_image, roi_box_pixels, roi_polygon_global_pixels)
        For polygon ROI, roi_image is masked + cropped to polygon bounding box.
        """
        polygon = self.resolve_roi_polygon_pixels(image)
        if polygon is not None:
            x, y, w, h = cv2.boundingRect(polygon)
            crop = image[y:y + h, x:x + w].copy()
            local_polygon = polygon - np.array([x, y], dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [local_polygon], 255)
            masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
            return masked_crop, (x, y, x + w, y + h), polygon

        box = self.resolve_roi_pixels(image)
        if box is None:
            return image, None, None
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2].copy(), box, None

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------

    def _run_yolo(self, image: np.ndarray) -> List[dict]:
        """Run workpiece YOLO on the (ROI-cropped) image."""
        if self._yolo is None:
            self._lazy_load_yolo()
        if self._yolo is None:
            return []
        try:
            results = self._yolo.predict(source=image, conf=0.3, verbose=False)
        except Exception as e:
            logger.info(f"[WorkspaceMonitor] YOLO inference failed: {e}")
            return []

        detections: List[dict] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                cls_name = self._yolo.names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append({
                    "class_name": cls_name,
                    "confidence": round(float(box.conf.item()), 3),
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                })
        return detections

    def _lazy_load_yolo(self):
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self._yolo_model_path)
            logger.info(f"[WorkspaceMonitor] YOLO loaded: {self._yolo_model_path}")
        except Exception as e:
            logger.info(f"[WorkspaceMonitor] YOLO load failed: {e}")

    # ------------------------------------------------------------------
    # VLM
    # ------------------------------------------------------------------

    def _encode_image(self, image: np.ndarray) -> str:
        """BGR numpy array → base64 JPEG string."""
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok, buf = cv2.imencode(".jpg", image, params)
        if not ok:
            raise RuntimeError("Image encoding failed")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _build_prompt(self, yolo_detections: List[dict]) -> str:
        assembly_list = ", ".join(sorted(KNOWN_ASSEMBLY_CLASSES))
        clutter_hint = "\n".join(f"  - {ex}" for ex in CLUTTER_EXAMPLES)

        if yolo_detections:
            yolo_lines = [
                f"  - {d['class_name']} (conf {d['confidence']:.0%}, center {d['center']})"
                for d in yolo_detections
            ]
            yolo_block = "YOLO detected the following known assembly parts in this frame:\n" + "\n".join(yolo_lines)
        else:
            yolo_block = "YOLO detected no known assembly parts in this frame."

        roi_note = (
            "Note: the image has already been cropped to the robot workspace region."
            if self.roi is not None
            else ""
        )

        return f"""You are an industrial robot workspace safety inspector.

[Task]
Analyse the image and determine whether any FOREIGN OBJECTS (clutter) are present on the workspace table.
Foreign objects are items unrelated to the assembly task that could interfere with robot arm movement or grasping accuracy.
{roi_note}

[Allowed assembly parts — do NOT flag these]
{assembly_list}

[YOLO detection context]
{yolo_block}

[Common clutter types to look for (not exhaustive)]
{clutter_hint}

[Response format — JSON only]
{{
  "has_clutter": true or false,
  "severity": "blocking" | "warning" | "safe",
  "detected_objects": ["object1", "object2", ...],
  "suggestion": "Short action for the operator (English, ≤15 words)"
}}

Severity definitions:
- "blocking": clutter clearly inside the operation zone; task must be paused
- "warning": clutter near the edge or a potential risk; proceed with caution
- "safe": workspace is clean, no foreign objects
"""

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            self._client = get_openai_client()
            return True
        except Exception as e:
            logger.info(f"[WorkspaceMonitor] Failed to create OpenAI client: {e}")
            return False

    def _call_vlm(
        self,
        image: np.ndarray,
        yolo_detections: List[dict],
        triggered_by: str,
    ) -> Optional[ClutterCheckResult]:
        if not self._ensure_client():
            return None

        try:
            b64 = self._encode_image(image)
        except Exception as e:
            logger.info(f"[WorkspaceMonitor] Image encoding failed: {e}")
            return None

        prompt = self._build_prompt(yolo_detections)

        try:
            logger.info(f"[WorkspaceMonitor] Calling {self.vlm_model} (trigger: {triggered_by})...")
            t0 = time.time()
            resp = self._client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "low",  # saves tokens; low = 512px tile
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.0,
            )
            elapsed = time.time() - t0

            raw = resp.choices[0].message.content
            data = json.loads(raw)

            result = ClutterCheckResult(
                has_clutter=bool(data.get("has_clutter", False)),
                severity=data.get("severity", "safe"),
                detected_objects=data.get("detected_objects", []),
                suggestion=data.get("suggestion", ""),
                triggered_by=triggered_by,
                raw_yolo_detections=yolo_detections,
            )

            tokens_used = getattr(resp.usage, "total_tokens", "?")
            logger.info(f"[WorkspaceMonitor] VLM response in {elapsed:.1f}s  (tokens: {tokens_used})")
            logger.info(f"  {result.summary()}")
            return result

        except json.JSONDecodeError as e:
            logger.info(f"[WorkspaceMonitor] JSON parse error: {e}  raw={raw!r}")
            return None
        except Exception as e:
            logger.info(f"[WorkspaceMonitor] VLM call failed: {e}")
            return None


def blackout_outside_roi_full_frame(
    image: np.ndarray,
    *,
    roi: Optional[Tuple[float, float, float, float]] = None,
    roi_is_ratio: bool = True,
    roi_polygon: Optional[List[Tuple[float, float]]] = None,
    roi_polygon_is_ratio: bool = False,
) -> np.ndarray:
    """
    Return a copy of ``image`` (BGR) with pixels **outside** the ROI set to black.

    Uses the same ROI rules as :class:`WorkspaceClutterMonitor` (polygon over rectangle).
    If no ROI is configured, returns ``image.copy()`` (full frame kept).
    """
    m = WorkspaceClutterMonitor(
        roi=roi,
        roi_is_ratio=roi_is_ratio,
        roi_polygon=roi_polygon,
        roi_polygon_is_ratio=roi_polygon_is_ratio,
    )
    return m.blackout_outside_roi(image)


# =============================================================================
# RealSense D435i capture helper
# =============================================================================

class RealSenseCapture:
    """
    Intel RealSense D435i capture helper.

    Initialisation priority (highest first):
      1. external_pipeline supplied — share with the main system (no resource conflict).
      2. serial_number provided, or env var D435I_SERIAL is set.
      3. Auto-scan connected devices for the first D435I.

    Usage:
        with RealSenseCapture() as cam:
            color, depth = cam.get_frame()
    or:
        cam = RealSenseCapture(serial_number="<D435I_SERIAL>")
        cam.start()
        color, depth = cam.get_frame()
        cam.stop()
    """

    def __init__(
        self,
        serial_number: Optional[str] = None,
        external_pipeline=None,
        external_align=None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        warmup_frames: int = 5,
    ):
        self._serial = serial_number or os.environ.get("D435I_SERIAL")
        self._external = external_pipeline is not None
        self._pipeline = external_pipeline
        self._align = external_align
        self._width = width
        self._height = height
        self._fps = fps
        self._warmup_frames = warmup_frames
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def start(self) -> bool:
        """Initialise and start the pipeline (skipped for external pipelines)."""
        if self._external:
            self._started = True
            return True
        try:
            import pyrealsense2 as rs
            serial = self._serial or self._find_d435i_serial()
            if serial is None:
                logger.error("[RealSenseCapture] ERROR: D435I not found. Check connection or set D435I_SERIAL in .env")
                return False

            logger.info(f"[RealSenseCapture] Using D435i serial: {serial}")
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
            cfg.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
            self._pipeline.start(cfg)
            self._align = rs.align(rs.stream.color)

            for _ in range(self._warmup_frames):
                self._pipeline.wait_for_frames(timeout_ms=2000)

            logger.info(f"[RealSenseCapture] Camera started ({self._width}x{self._height} @ {self._fps}fps)")
            self._started = True
            return True
        except Exception as e:
            logger.info(f"[RealSenseCapture] Start failed: {e}")
            return False

    def stop(self):
        """Stop the pipeline (no-op for external pipelines)."""
        if not self._external and self._pipeline and self._started:
            try:
                self._pipeline.stop()
                logger.info("[RealSenseCapture] Pipeline stopped.")
            except Exception:
                pass
        self._started = False

    def get_frame(self, timeout_ms: int = 2000):
        """
        Capture one frame.

        Returns:
            (color_bgr: np.ndarray, depth_frame) or (None, None) on failure.
            depth_frame is rs.depth_frame — use .get_distance(x, y) for depth values.
        """
        if not self._pipeline:
            logger.error("[RealSenseCapture] ERROR: pipeline not initialised, call start() first")
            return None, None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
            if self._align:
                frames = self._align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame:
                return None, None
            return np.asanyarray(color_frame.get_data()), depth_frame
        except Exception as e:
            logger.info(f"[RealSenseCapture] Capture failed: {e}")
            return None, None

    def capture_color(self, timeout_ms: int = 2000) -> Optional[np.ndarray]:
        """Convenience: capture colour frame only."""
        color, _ = self.get_frame(timeout_ms=timeout_ms)
        return color

    @staticmethod
    def _find_d435i_serial() -> Optional[str]:
        """Scan connected RealSense devices and return the first D435I serial."""
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            for dev in ctx.query_devices():
                name = dev.get_info(rs.camera_info.name).upper()
                if "D435I" in name or "D435 I" in name:
                    serial = dev.get_info(rs.camera_info.serial_number)
                    logger.info(f"[RealSenseCapture] Auto-detected D435I: {serial}")
                    return serial
            logger.warning("[RealSenseCapture] WARNING: No D435I device found")
            return None
        except ImportError:
            logger.warning("[RealSenseCapture] WARNING: pyrealsense2 not installed")
            return None

    @staticmethod
    def list_devices():
        """Print all connected RealSense devices (for debugging)."""
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = list(ctx.query_devices())
            if not devices:
                logger.info("[RealSenseCapture] No RealSense devices detected")
                return
            for dev in devices:
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                logger.info(f"  {name}  serial: {serial}")
        except Exception as e:
            logger.info(f"[RealSenseCapture] list_devices failed: {e}")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Workspace clutter detector — GPT-4o Vision + YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workspace_monitor.py desk.jpg
  python workspace_monitor.py desk.jpg --roi 0.05 0.1 0.95 0.9
  python workspace_monitor.py desk.jpg --model gpt-4o-mini
  python workspace_monitor.py --camera
  python -m ws_racr.execution.safety.workspace_monitor --camera --serial <D435I_SERIAL>
  python workspace_monitor.py --camera --roi 0.05 0.1 0.95 0.9 --save result.jpg
  python workspace_monitor.py --list-cameras
  python workspace_monitor.py --camera --roi 0.05 0.1 0.95 0.9 --triple ./out/snap --skip-check
        """,
    )
    parser.add_argument("image", nargs="?", help="Path to image file (.jpg/.png)")
    parser.add_argument("--camera", action="store_true", help="Capture from RealSense D435i")
    parser.add_argument("--serial", default=None,
                        help="D435i serial number (overrides env D435I_SERIAL)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List all connected RealSense devices and exit")
    parser.add_argument("--roi", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                        default=None,
                        help="ROI as ratio fractions 0-1, e.g. --roi 0.05 0.1 0.95 0.9")
    parser.add_argument("--roi-pixels", action="store_true",
                        help="Treat --roi values as absolute pixel coordinates")
    parser.add_argument(
        "--roi-points",
        nargs=8,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
        default=None,
        help="Polygon ROI with 4 points. Example: --roi-points 0 426 1280 369 1280 0 0 0",
    )
    parser.add_argument(
        "--roi-points-ratio",
        action="store_true",
        help="Treat --roi-points values as 0-1 ratios instead of pixels",
    )
    parser.add_argument("--model", default="gpt-4o",
                        help="Vision model name (default: gpt-4o)")
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="Cooldown in seconds (set 0 for testing, default: 0)")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Disable YOLO pre-filter (pure VLM mode)")
    parser.add_argument("--save", default=None,
                        help="Save annotated image to this path")
    parser.add_argument(
        "--triple",
        default=None,
        metavar="PREFIX",
        help="Save PREFIX_01_raw.jpg, PREFIX_02_roi_black.jpg, PREFIX_03_clutter_overlay.jpg "
             "(03: blackout + YOLO + English hints; full clutter text needs VLM)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip VLM; with --triple still runs YOLO and writes 03 (YOLO-only preview)",
    )
    parser.add_argument("--no-display", action="store_true",
                        help="Do not open OpenCV window")
    return parser.parse_args()


def _draw_result(
    base_bgr: np.ndarray,
    result: ClutterCheckResult,
    roi_box: Optional[Tuple[int, int, int, int]] = None,
    roi_polygon: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Overlay ROI outline, YOLO boxes, and English status lines on ``base_bgr``."""
    vis = base_bgr.copy()
    severity_color = {"blocking": (0, 0, 255), "warning": (0, 165, 255), "safe": (0, 200, 0)}
    color = severity_color.get(result.severity, (128, 128, 128))

    # Draw ROI polygon first when provided
    if roi_polygon is not None:
        cv2.polylines(vis, [roi_polygon], isClosed=True, color=(255, 255, 255), thickness=3)
        cv2.polylines(vis, [roi_polygon], isClosed=True, color=(0, 180, 255), thickness=1)
        p0 = tuple(roi_polygon[0].tolist())
        cv2.putText(vis, "ROI", (p0[0] + 6, p0[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)

    # Draw ROI rectangle (white + orange)
    if roi_box is not None:
        rx1, ry1, rx2, ry2 = roi_box
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 255), 3)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 180, 255), 1)
        cv2.putText(vis, "ROI", (rx1 + 4, ry1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)

    # Draw YOLO bounding boxes (orange — known assembly parts)
    yolo_offset_x = roi_box[0] if roi_box else 0
    yolo_offset_y = roi_box[1] if roi_box else 0
    for det in result.raw_yolo_detections:
        x1, y1, x2, y2 = det["bbox"]
        x1 += yolo_offset_x; y1 += yolo_offset_y
        x2 += yolo_offset_x; y2 += yolo_offset_y
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    # English status (line 1) + short English action hint (line 2)
    status_text = result.summary()
    cv2.putText(vis, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)   # outline
    cv2.putText(vis, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)        # colored text

    if result.has_clutter:
        if result.detected_objects:
            hint2 = "Please move away: " + ", ".join(result.detected_objects)
        else:
            hint2 = "Please remove foreign objects from the work area."
    else:
        hint2 = "Workspace clear — safe to proceed."
    y2 = 62
    cv2.putText(vis, hint2, (10, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
    cv2.putText(vis, hint2, (10, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

    return vis


def _triple_paths(prefix: str) -> Tuple[str, str, str]:
    base = Path(prefix).expanduser()
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = str(base)
    return (
        stem + "_01_raw.jpg",
        stem + "_02_roi_black.jpg",
        stem + "_03_clutter_overlay.jpg",
    )


def main():
    args = _parse_args()

    if args.list_cameras:
        logger.info("=== Connected RealSense devices ===")
        RealSenseCapture.list_devices()
        sys.exit(0)

    # --- Acquire image ---
    if args.image:
        path = Path(args.image)
        if not path.exists():
            logger.error(f"ERROR: image not found: {path}")
            sys.exit(1)
        image = cv2.imread(str(path))
        if image is None:
            logger.error(f"ERROR: could not read image: {path}")
            sys.exit(1)
        logger.info(f"Image loaded: {path}  ({image.shape[1]}x{image.shape[0]})")

    elif args.camera:
        cam = RealSenseCapture(serial_number=args.serial)
        if not cam.start():
            sys.exit(1)
        logger.info("Capturing frame from RealSense D435i...")
        image = cam.capture_color()
        cam.stop()
        if image is None:
            logger.error("ERROR: capture failed")
            sys.exit(1)
        logger.info(f"Captured: {image.shape[1]}x{image.shape[0]}")

    else:
        logger.error("ERROR: specify an image path or use --camera")
        logger.info("       python workspace_monitor.py --help")
        sys.exit(1)

    # --- Build monitor ---
    roi = tuple(args.roi) if args.roi else None
    roi_polygon = None
    if args.roi_points:
        raw = args.roi_points
        roi_polygon = [
            (raw[0], raw[1]),
            (raw[2], raw[3]),
            (raw[4], raw[5]),
            (raw[6], raw[7]),
        ]

    monitor = WorkspaceClutterMonitor(
        vlm_model=args.model,
        cooldown_seconds=args.cooldown,
        roi=roi,
        roi_is_ratio=not args.roi_pixels,
        roi_polygon=roi_polygon,
        roi_polygon_is_ratio=args.roi_points_ratio,
    )

    triple_p1 = triple_p2 = triple_p3 = None
    img_blackout: Optional[np.ndarray] = None
    if args.triple:
        triple_p1, triple_p2, triple_p3 = _triple_paths(args.triple)
        cv2.imwrite(triple_p1, image)
        img_blackout = monitor.blackout_outside_roi(image)
        cv2.imwrite(triple_p2, img_blackout)
        logger.info(f"Triple [1] raw:     {triple_p1}\n      [2] blackout: {triple_p2}")

    if args.skip_check:
        if args.triple and img_blackout is not None and triple_p3 is not None:
            if args.no_yolo:
                preview = ClutterCheckResult(
                    has_clutter=False,
                    severity="safe",
                    detected_objects=[],
                    suggestion="",
                    triggered_by="triple_preview",
                    raw_yolo_detections=[],
                )
            else:
                roi_image, roi_box, roi_polygon = monitor._apply_roi(image)
                monitor._last_roi_box = roi_box
                monitor._last_roi_polygon = roi_polygon
                dets = monitor._run_yolo(roi_image)
                preview = ClutterCheckResult(
                    has_clutter=False,
                    severity="safe",
                    detected_objects=[],
                    suggestion="(VLM not called)",
                    triggered_by="triple_preview",
                    raw_yolo_detections=dets,
                )
            vis3 = _draw_result(
                img_blackout,
                preview,
                roi_box=monitor.last_roi_box,
                roi_polygon=monitor.last_roi_polygon,
            )
            cv2.imwrite(triple_p3, vis3)
            logger.info(f"      [3] overlay (YOLO-only preview, no VLM): {triple_p3}")
        logger.info("--skip-check: exiting without VLM clutter check.")
        sys.exit(0)

    if args.no_yolo:
        monitor._yolo = type("_Dummy", (), {"predict": lambda *a, **k: [], "names": {}})()
        logger.info("YOLO disabled — pure VLM mode")

    # --- Run check ---
    logger.info("\n" + "=" * 60)
    result = monitor.check(image, triggered_by="cli_test")
    logger.info("=" * 60 + "\n")

    if result is None:
        logger.error("ERROR: check returned None (API key issue or cooldown)")
        sys.exit(1)

    if args.triple and img_blackout is not None and triple_p3 is not None:
        vis3 = _draw_result(
            img_blackout,
            result,
            roi_box=monitor.last_roi_box,
            roi_polygon=monitor.last_roi_polygon,
        )
        cv2.imwrite(triple_p3, vis3)
        logger.info(f"Triple [3] clutter overlay (blackout + YOLO + hints): {triple_p3}")

    # --- Visualise / save ---
    vis = _draw_result(
        image,
        result,
        roi_box=monitor.last_roi_box,
        roi_polygon=monitor.last_roi_polygon,
    )

    if args.save:
        cv2.imwrite(args.save, vis)
        logger.info(f"Annotated image saved: {args.save}")

    if not args.no_display:
        try:
            cv2.imshow("WorkspaceMonitor", vis)
            logger.info("(Press any key to close window)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass  # headless environment

    # Exit codes: safe=0, warning=1, blocking=2
    sys.exit({"safe": 0, "warning": 1, "blocking": 2}.get(result.severity, 0))


if __name__ == "__main__":
    main()

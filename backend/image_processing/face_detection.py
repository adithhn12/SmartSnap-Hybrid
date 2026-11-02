"""Consensus face detection and visibility estimation for human-focused photos."""

import logging
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
FACE_CASCADE_FILES = {
    "default": "haarcascade_frontalface_default.xml",
    "alt": "haarcascade_frontalface_alt.xml",
    "alt2": "haarcascade_frontalface_alt2.xml",
}
FACE_CONSENSUS_THRESHOLD = 2 / 3
DETECTION_STABILITY_VARIANTS = ("gray", "equalized")


class FaceDetector:
    def __init__(self):
        self.cascades: Dict[str, Any] = {}
        cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if cascade_dir:
            for name, filename in FACE_CASCADE_FILES.items():
                classifier = cv2.CascadeClassifier(cascade_dir + filename)
                if not classifier.empty():
                    self.cascades[name] = classifier

    @staticmethod
    def _intersection_over_union(
        first: Tuple[int, int, int, int],
        second: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[0] + first[2], second[0] + second[2])
        y2 = min(first[1] + first[3], second[1] + second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = first[2] * first[3] + second[2] * second[3] - intersection
        return intersection / float(union) if union > 0 else 0.0

    def _collect_detections(
        self, gray: np.ndarray, min_size: int
    ) -> List[Tuple[Tuple[int, int, int, int], str, str]]:
        detections = []
        variants = {
            "gray": gray,
            "equalized": cv2.equalizeHist(gray),
        }
        for variant_name, variant in variants.items():
            for cascade_name, classifier in self.cascades.items():
                try:
                    rectangles = classifier.detectMultiScale(
                        variant,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(min_size, min_size),
                    )
                except cv2.error as error:
                    LOGGER.warning(
                        "Face cascade %s failed on %s image: %s",
                        cascade_name,
                        variant_name,
                        error,
                    )
                    continue
                for x, y, width, height in rectangles:
                    aspect_ratio = width / float(height) if height else 0.0
                    if not 0.75 <= aspect_ratio <= 1.33:
                        continue
                    detections.append(
                        (
                            (int(x), int(y), int(width), int(height)),
                            cascade_name,
                            variant_name,
                        )
                    )
        return detections

    def _verified_candidates(
        self, detections: List[Tuple[Tuple[int, int, int, int], str, str]]
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        clusters: List[List[Tuple[Tuple[int, int, int, int], str, str]]] = []
        for detection in sorted(
            detections, key=lambda item: item[0][2] * item[0][3], reverse=True
        ):
            box = detection[0]
            matching_cluster = next(
                (
                    cluster
                    for cluster in clusters
                    if max(
                        self._intersection_over_union(box, candidate[0])
                        for candidate in cluster
                    )
                    >= 0.30
                ),
                None,
            )
            if matching_cluster is None:
                clusters.append([detection])
            else:
                matching_cluster.append(detection)

        verified = []
        for cluster in clusters:
            models_by_variant: Dict[str, Set[str]] = {
                variant: set() for variant in DETECTION_STABILITY_VARIANTS
            }
            for _, cascade_name, variant_name in cluster:
                models_by_variant[variant_name].add(cascade_name)
            if not all(
                len(models_by_variant[variant]) / len(FACE_CASCADE_FILES)
                >= FACE_CONSENSUS_THRESHOLD
                for variant in DETECTION_STABILITY_VARIANTS
            ):
                continue

            boxes = np.asarray([item[0] for item in cluster], dtype=np.float64)
            median_box = np.median(boxes, axis=0)
            box = tuple(int(round(value)) for value in median_box)
            confidence = float(
                np.mean(
                    [
                        len(models_by_variant[variant])
                        / len(FACE_CASCADE_FILES)
                        for variant in DETECTION_STABILITY_VARIANTS
                    ]
                )
            )
            verified.append((box, confidence))
        return verified

    def detect_and_evaluate(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        if not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            raise ValueError("Expected a non-empty image for face detection.")
        original_gray = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            if image_bgr.ndim == 3
            else image_bgr.copy()
        )
        gray = original_gray
        original_height, original_width = gray.shape[:2]
        if not self.cascades:
            return self._empty_result("No OpenCV frontal-face cascades are available")

        # Bound work for full-resolution phone images while keeping small faces visible.
        max_dimension = max(original_height, original_width)
        scale = min(1.0, 1600.0 / max_dimension)
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        height, width = gray.shape[:2]
        min_size = max(32, int(round(min(height, width) * 0.05)))
        detections = self._collect_detections(gray, min_size)
        verified = self._verified_candidates(detections)
        verified.sort(key=lambda item: item[0][2] * item[0][3], reverse=True)

        scale_x = original_width / float(width)
        scale_y = original_height / float(height)
        face_boxes: List[Dict[str, int]] = []
        clarity_scores = []
        total_area = 0.0
        highest_confidence = max((confidence for _, confidence in verified), default=0.0)
        for (x, y, face_width, face_height), _ in verified:
            x = max(0, min(original_width - 1, int(round(x * scale_x))))
            y = max(0, min(original_height - 1, int(round(y * scale_y))))
            face_width = min(original_width - x, int(round(face_width * scale_x)))
            face_height = min(original_height - y, int(round(face_height * scale_y)))
            if face_width <= 0 or face_height <= 0:
                continue
            face_boxes.append(
                {"x": x, "y": y, "w": face_width, "h": face_height}
            )
            total_area += face_width * face_height
            face_roi = original_gray[
                y : y + face_height, x : x + face_width
            ]
            if face_roi.size:
                standardized_face = cv2.resize(
                    face_roi,
                    (160, 160),
                    interpolation=(
                        cv2.INTER_AREA
                        if face_width > 160 or face_height > 160
                        else cv2.INTER_LINEAR
                    ),
                )
                regional_detail = []
                for row in range(4):
                    for column in range(4):
                        tile = standardized_face[
                            row * 40 : (row + 1) * 40,
                            column * 40 : (column + 1) * 40,
                        ]
                        regional_detail.append(
                            float(cv2.Laplacian(tile, cv2.CV_64F).var())
                        )
                # A lower-quartile regional measure is less inflated by a single
                # sharp occlusion edge than whole-face Laplacian variance.
                clarity = float(np.quantile(regional_detail, 0.25))
                clarity_scores.append(1.0 - np.exp(-clarity / 150.0))

        if not face_boxes:
            return self._empty_result(
                "No stable face consensus; unverified Haar proposals are discarded"
            )

        area_coverage = total_area / float(original_height * original_width)
        prominence = float(np.clip(area_coverage / 0.10, 0.2, 1.0))
        clarity = float(np.mean(clarity_scores)) if clarity_scores else 0.5
        visibility = float(
            np.round(np.clip((0.35 + 0.35 * prominence + 0.30 * clarity) * 100, 0, 100), 2)
        )
        description = (
            "Poor (Face detected but small, distant, or indistinct)"
            if visibility < 40
            else "Acceptable (Face visible with recognizable features)"
            if visibility <= 75
            else "Clear (Crisp, well-framed face with prominent detail)"
        )
        confidence = float(np.round(highest_confidence, 3))
        method = (
            "Haar consensus (2/3 independent face cascades, stable on grayscale "
            "and histogram-equalized images)"
        )
        return {
            "face_visibility": visibility,
            "face_count": len(face_boxes),
            "faces": face_boxes,
            "area_coverage_pct": float(np.round(area_coverage * 100.0, 2)),
            "confidence": confidence,
            "method": method,
            "description": description,
            "explanation": (
                f"Detected {len(face_boxes)} face(s) with consensus confidence "
                f"{confidence:.2f} using {method}; unverified cascade proposals are omitted."
            ),
        }

    @staticmethod
    def _empty_result(reason: str) -> Dict[str, Any]:
        return {
            "face_visibility": 0.0,
            "face_count": 0,
            "faces": [],
            "area_coverage_pct": 0.0,
            "confidence": 0.0,
            "method": reason,
            "description": "No verified faces detected",
            "explanation": (
                f"{reason}. A Haar bounding box alone is not treated as a face."
            ),
        }


_detector = None


def estimate_face_visibility(image_bgr: np.ndarray) -> Dict[str, Any]:
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector.detect_and_evaluate(image_bgr)

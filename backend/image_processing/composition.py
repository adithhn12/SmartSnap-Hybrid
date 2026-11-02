"""
Composition Assessment Module for SmartSnap.
Evaluates photographic framing considering:
1. Rule-of-Thirds alignment (intersections at 1/3, 2/3).
2. Intentional Center Framing (especially valid and desirable in portraits, symmetry, and architectural shots).
3. Border boundary safety margin (penalizing accidental edge cutoff).
"""

from typing import Dict, Any, Optional, List
import cv2
import numpy as np


def estimate_composition(
    image_bgr: np.ndarray,
    faces: Optional[List[Dict[str, int]]] = None,
    category: str = "General Photography",
) -> Dict[str, Any]:
    """
    Evaluates photographic composition heuristics without unfairly penalizing centered subjects.
    """
    category_key = category.casefold()
    h, w = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr.copy()

    if category_key == "document":
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        page_quad = None
        image_area = float(h * w)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if (
                len(approx) == 4
                and cv2.isContourConvex(approx)
                and cv2.contourArea(approx) >= 0.15 * image_area
            ):
                page_quad = approx.reshape(4, 2).astype(np.float32)
                break

        if page_quad is None:
            return {
                "composition": 50.0,
                "subject_type": "Document page not detected",
                "subject_center": {"x": 0.5, "y": 0.5},
                "min_anchor_distance": None,
                "min_border_distance": None,
                "method": "Document page quadrilateral not detected; neutral score",
                "description": "Neutral (page boundaries could not be established)",
                "explanation": "No sufficiently large convex page outline was detected; document framing was not penalized.",
            }

        center = np.mean(page_quad, axis=0)
        center_offset = float(
            np.linalg.norm(center / np.array([w, h], dtype=np.float32) - 0.5)
        )
        edges_xy = np.roll(page_quad, -1, axis=0) - page_quad
        edge_lengths = np.linalg.norm(edges_xy, axis=1)
        opposite_width = 0.5 * (edge_lengths[0] + edge_lengths[2])
        opposite_height = 0.5 * (edge_lengths[1] + edge_lengths[3])
        aspect_imbalance = abs(
            edge_lengths[0] - edge_lengths[2]
        ) / max(opposite_width, 1.0) + abs(
            edge_lengths[1] - edge_lengths[3]
        ) / max(opposite_height, 1.0)
        score = 95.0 - 50.0 * center_offset - 15.0 * aspect_imbalance
        composition_score = float(np.round(np.clip(score, 15.0, 95.0), 2))
        return {
            "composition": composition_score,
            "subject_type": "Detected document page",
            "subject_center": {
                "x": float(np.round(center[0] / w, 3)),
                "y": float(np.round(center[1] / h, 3)),
            },
            "min_anchor_distance": None,
            "min_border_distance": None,
            "method": "Convex page quadrilateral center and opposite-edge consistency",
            "description": "Document page framing from detected page geometry",
            "explanation": (
                f"Detected page center offset is {center_offset:.3f}; "
                f"opposite-edge imbalance is {aspect_imbalance:.3f}."
            ),
        }

    # Face detections and portrait-head anchors apply only to human-focused intent.
    composition_faces = (
        faces or [] if category_key in {"portrait", "group photograph"} else []
    )

    # Step 1: Identify Subject Center (relative coordinates 0.0 to 1.0)
    subject_type = ""
    group_bounds = None
    if composition_faces and category_key == "group photograph":
        centers = np.array([
            ((face["x"] + face["w"] / 2.0) / float(w),
             (face["y"] + face["h"] / 2.0) / float(h))
            for face in composition_faces
        ])
        cx, cy = np.mean(centers, axis=0)
        group_bounds = (
            min(face["x"] for face in composition_faces) / float(w),
            min(face["y"] for face in composition_faces) / float(h),
            max(face["x"] + face["w"] for face in composition_faces) / float(w),
            max(face["y"] + face["h"] for face in composition_faces) / float(h),
        )
        subject_type = "Detected group centroid"
    elif composition_faces:
        # Use largest detected face as primary subject
        largest_face = max(composition_faces, key=lambda f: f.get("w", 0) * f.get("h", 0))
        cx = (largest_face["x"] + largest_face["w"] / 2.0) / float(w)
        cy = (largest_face["y"] + largest_face["h"] / 2.0) / float(h)
        subject_type = "Detected Face"
    else:
        # Heuristic visual saliency: Sobel gradient energy centroid
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        _, thresh = cv2.threshold(mag, np.mean(mag) * 1.5, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(thresh)
        if moments["m00"] > 0:
            cx = (moments["m10"] / moments["m00"]) / float(w)
            cy = (moments["m01"] / moments["m00"]) / float(h)
            subject_type = "Visual Saliency Centroid"
        else:
            cx, cy = 0.5, 0.5
            subject_type = "Default Frame Center"

    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))

    # Step 2: Use portrait head placement only for human-focused categories.
    anchors = [
        (1.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
        (0.50, 0.50),
    ]
    if category_key in {"portrait", "group photograph"}:
        anchors.append((0.50, 0.45))

    min_dist = min(np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) for (ax, ay) in anchors)
    # Max anchor distance in normal framing is ~0.18
    framing_score = float(np.clip(1.0 - (min_dist / 0.22), 0.35, 1.0))

    # Step 3: Boundary Safety Margin
    # Penalize if subject is awkwardly clipped against image borders (< 0.05)
    if group_bounds is not None:
        min_edge_dist = min(group_bounds[0], group_bounds[1], 1.0 - group_bounds[2], 1.0 - group_bounds[3])
    else:
        edge_dist_x = min(cx, 1.0 - cx)
        edge_dist_y = min(cy, 1.0 - cy)
        min_edge_dist = min(edge_dist_x, edge_dist_y)

    if min_edge_dist < 0.04:
        border_penalty = 0.55  # severely clipped
    elif min_edge_dist < 0.08:
        border_penalty = 0.80  # slightly close to edge
    else:
        border_penalty = 1.00  # comfortable breathing room

    # Combined score
    composite = (0.75 * framing_score + 0.25) * border_penalty * 100.0
    composition_score = float(np.round(np.clip(composite, 15.0, 95.0), 2))

    if composition_score < 45.0:
        desc = "Poor (Subject placed too close to border or awkward framing)"
    elif composition_score <= 75.0:
        desc = "Average (Acceptable framing, slight deviation from ideal grid)"
    else:
        desc = "Good (Balanced alignment adhering to Rule of Thirds or Center Framing with safe margins)"

    return {
        "composition": composition_score,
        "subject_type": subject_type,
        "subject_center": {"x": float(np.round(cx, 3)), "y": float(np.round(cy, 3))},
        "min_anchor_distance": float(np.round(min_dist, 3)),
        "min_border_distance": float(np.round(min_edge_dist, 3)),
        "method": "Rule-of-Thirds & Center Framing anchor distance with border safety margin",
        "description": desc,
        "explanation": f"Subject localized via '{subject_type}' at ({cx:.2f}, {cy:.2f}). Framing alignment: {framing_score * 100:.1f}%, Border breathing margin: {min_edge_dist:.2f}.",
    }

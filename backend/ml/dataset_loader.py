"""KonIQ-10k image/MOS matching and dataset validation."""

import csv
import math
from pathlib import Path, PurePosixPath
from typing import Dict, List, Tuple


class DatasetValidationError(ValueError):
    """Raised when the supplied dataset does not have the required structure."""


def _find_dataset_paths(dataset_root: Path) -> Tuple[Path, Path]:
    layouts = (
        (dataset_root / "512x384", dataset_root / "annotations"),
        (dataset_root / "KonIQ-10k" / "512x384", dataset_root / "KonIQ-10k" / "annotations"),
    )
    for image_dir, annotations_dir in layouts:
        if image_dir.is_dir() and annotations_dir.is_dir():
            return image_dir, annotations_dir
    expected = "512x384/ and annotations/ (or those two folders under KonIQ-10k/)"
    raise DatasetValidationError(f"Dataset folders not found; expected {expected} at {dataset_root}.")


def load_koniq_records(dataset_root: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Match score rows to local images by image_name; indicators are not model inputs."""
    root = Path(dataset_root).expanduser().resolve()
    image_dir, annotations_dir = _find_dataset_paths(root)
    scores_path = annotations_dir / "koniq10k_scores_and_distributions.csv"
    if not scores_path.is_file():
        raise DatasetValidationError(f"Required MOS annotation CSV is missing: {scores_path}")

    images: Dict[str, Path] = {}
    duplicate_images: List[str] = []
    for image_path in image_dir.iterdir():
        if not image_path.is_file():
            continue
        key = image_path.name.casefold()
        if key in images:
            duplicate_images.append(image_path.name)
        else:
            images[key] = image_path
    if duplicate_images:
        raise DatasetValidationError(f"Duplicate image filenames found: {duplicate_images[:5]}")

    records: List[Dict[str, object]] = []
    seen_annotations = set()
    annotated_image_keys = set()
    unmatched_annotations: List[str] = []
    invalid_mos: List[str] = []
    duplicate_annotations: List[str] = []
    missing_image_names: List[str] = []
    annotation_rows = 0

    with scores_path.open("r", encoding="utf-8-sig", newline="") as annotation_file:
        reader = csv.DictReader(annotation_file)
        if not reader.fieldnames or not {"image_name", "MOS"}.issubset(reader.fieldnames):
            raise DatasetValidationError(
                f"{scores_path.name} must contain image_name and MOS columns; "
                f"found {reader.fieldnames}."
            )
        for row_number, row in enumerate(reader, start=2):
            annotation_rows += 1
            raw_name = (row.get("image_name") or "").strip().replace("\\", "/")
            image_name = PurePosixPath(raw_name).name
            if not image_name:
                missing_image_names.append(f"row {row_number}")
                continue
            key = image_name.casefold()
            if key in seen_annotations:
                duplicate_annotations.append(image_name)
                continue
            seen_annotations.add(key)
            image_path = images.get(key)
            if image_path is None:
                unmatched_annotations.append(image_name)
                continue
            annotated_image_keys.add(key)
            try:
                mos = float(row.get("MOS", ""))
            except (TypeError, ValueError):
                invalid_mos.append(f"{image_name}: missing or non-numeric MOS")
                continue
            if not math.isfinite(mos) or not 1.0 <= mos <= 5.0:
                invalid_mos.append(f"{image_name}: MOS outside the expected [1, 5] range")
                continue
            records.append({"image_name": image_path.name, "image_path": image_path, "mos": mos})

    unannotated_images = [
        path.name for key, path in images.items() if key not in annotated_image_keys
    ]
    report: Dict[str, object] = {
        "image_directory": str(image_dir.relative_to(root)),
        "scores_csv": str(scores_path.relative_to(root)),
        "images_found": len(images),
        "annotation_rows": annotation_rows,
        "matched_records": len(records),
        "image_annotation_matches": len(annotated_image_keys),
        "unmatched_annotations": unmatched_annotations,
        "unannotated_images": unannotated_images,
        "invalid_mos": invalid_mos,
        "missing_image_names": missing_image_names,
        "duplicate_annotations": duplicate_annotations,
    }
    if not records:
        raise DatasetValidationError(f"No valid image/MOS pairs found. Dataset validation: {report}")
    return records, report

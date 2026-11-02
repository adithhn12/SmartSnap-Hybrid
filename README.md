# SmartSnap V2 — Hybrid Image Quality Assessment

SmartSnap Hybrid combines OpenCV analysis and a Mamdani fuzzy decision engine with the preserved Random Forest regressor. For an uploaded image, the Random Forest estimates a human-perceived KonIQ mean opinion score (MOS); Mamdani inference combines that estimate with CV evidence and category-appropriate context to produce a 0–100 score and a `KEEP`, `RECOVER`, or `RETAKE` recommendation.

The four supported categories are **Portrait Photography**, **Item Photography**, **Landscape Photography**, and **General Photography**. Portrait assessment uses detected face regions and visibility; item assessment uses a lightweight OpenCV foreground proposal for object sharpness, exposure, and framing. Landscape uses regional scene sharpness, while General assesses global image evidence without assuming a subject type.

KonIQ MOS is a perceptual quality target, not a label for whether an image should be kept, edited, or retaken. The application’s final recommendation is a separately designed fuzzy decision.

## Architecture

```text
Uploaded image
  ├─ Existing CV analysis ──> blur, exposure, contrast, noise, face/object, composition diagnostics
  └─ Shared model extractor ──> brightness, contrast, colorfulness, sharpness, noise
                                 └─ Random Forest ──> predicted MOS [1, 5]
                                                       └─ Mamdani FIS
                                                          MOS + severe-blur evidence + category context
                                                          min antecedents/implication
                                                          max aggregation
                                                          centroid defuzzification
                                                          └─ final score + recommendation
```

The hybrid FIS uses the learned MOS plus direct blur/noise evidence and category-aware subject/composition evidence. Blur is an explicit, interpretable severe-defect signal because the regressor can smooth over a single severe defect. Noise has separate moderate- and high-defect rules because high estimated sensor noise can also be masked by a regression average. Face evidence is used only for Portrait Photography; object evidence is used only for Item Photography. Landscape and General Photography omit both subject-specific inputs. The original CV measurements and CV-only Mamdani rules remain available to the analysis response and simulation playground.

### Project modules

```text
backend/
  app.py                         FastAPI endpoints and one-time service initialization
  image_processing/              Existing OpenCV feature extractors
  fuzzy/                         Original Mamdani engine plus a separate hybrid rulebase
  ml/
    feature_extractor.py         Shared training and inference feature extraction
    dataset_loader.py            Dataset validation and image_name/MOS matching
    train_model.py               Split, validation selection, fit, test, and save
    evaluate_model.py            MAE, MSE, RMSE, R², and SRCC
    predictor.py                 Validated joblib loading and inference
    random_forest_model.joblib   Trained artifact (created by training)
    training_results.json        Dataset report, split, parameters, and measured metrics
  services/analysis_service.py  CV → Random Forest → Mamdani service integration
  tests/                         Existing suite and hybrid tests
frontend/src/pages/              Upload preview and hybrid results display
```

## Technologies and features

- Python, FastAPI, OpenCV, NumPy, scikit-learn, joblib, scikit-fuzzy
- React, Vite, and Recharts
- Random Forest input features, in fixed order:
  1. **Brightness** — whole-image grayscale luminance score, [0, 100].
  2. **Contrast** — existing RMS/percentile contrast score, [0, 100].
  3. **Colorfulness** — Hasler–Süsstrunk opponent-colorfulness statistic on 8-bit BGR channels.
  4. **Sharpness** — whole-image General Photography sharpness score, [0, 100], using the feature preprocessing retained by the trained artifact.
  5. **Noise** — existing texture-suppressed Immerkaer noise score, [0, 100].
- Regression target: `MOS` from the score/distribution CSV, on the original [1, 5] scale.

The model does not use image names/IDs, `MOS_zscore`, opinion distributions, `deep_feature`, or any other annotation-derived column. Training and inference both use the same whole-image extractor, without face-dependent input changes; this keeps the RF predictors aligned with global image MOS. Separate face- and category-aware CV measures remain interpretable fuzzy context. The same extractor and feature order are validated when loading a model and used for every prediction. Tree models do not require numeric feature scaling; these five values retain their documented native ranges. Invalid images are skipped and reported during training, and non-finite inputs fail explicitly at prediction; there is no imputation.

The face detector treats Haar boxes as proposals, not detections. It runs only for Portrait Photography; other categories report face visibility as not assessed. For enabled categories, it requires a matching box from at least two of three OpenCV frontal-face cascades, stable on both grayscale and histogram-equalized inputs. Candidate boxes must be at least `max(32 px, 5% of the smaller normalized image dimension)` and have a width/height ratio in `[0.75, 1.33]`; image processing is capped at a 1,600-pixel maximum side. Consensus is the reported confidence (`≥2/3`). A lower-quartile local detail measure reduces the chance that one sharp occlusion boundary inflates whole-face clarity. A skin-tone fallback is deliberately not used to label a face, since skin color is neither necessary nor sufficient evidence. This conservative frontal-face heuristic can miss profiles, heavily occluded faces, or tiny/distant subjects.

Item Photography uses OpenCV GrabCut with a central framing prior to propose a foreground region; it is not semantic object recognition. An ambiguous boundary returns neutral visibility and falls back to global/regional sharpness instead of treating segmentation failure as a defect. It can miss off-center subjects or merge an object with a cluttered background. No heavyweight object detector is introduced.

The CV blur metric used for category-aware diagnostics and FIS evidence subtracts estimated sensor-noise power from Laplacian variance before converting high-frequency energy into sharpness. For the 4-neighbor Laplacian, the independent white-noise gain is 20, so the subtracted component is `20 × sigma²`, floored at zero. The RF feature extractor explicitly keeps the prior uncorrected whole-image sharpness preprocessing used to train the loaded model; training and inference continue to use that same extractor. This separates the recommendation-oriented noise-corrected blur estimate from the deployed model’s established feature definition. Noise remains an estimate from lower-gradient image patches, not a semantic classifier, so highly compressed or unusually textured images can still confound it.

## KonIQ-10k dataset

Place the locally obtained dataset in either supported layout:

```text
datasets/
  512x384/                         # actual layout currently present in this workspace
  annotations/
    koniq10k_scores_and_distributions.csv
    koniq10k_indicators.csv
```

or:

```text
datasets/KonIQ-10k/
  512x384/
  annotations/
```

The loader uses the `image_name` and `MOS` columns from `koniq10k_scores_and_distributions.csv` and matches exact image basenames case-insensitively. It validates MOS as finite and within [1, 5], reports unmatched annotation rows and images, and skips unreadable matched files with their filenames recorded in the results report. It does not join `koniq10k_indicators.csv`: its `image_id` is not the score file’s `image_name` identifier, and the pixel features are recomputed from images instead. Dataset files and archives are excluded by `.gitignore`; do not commit them.

Dataset attribution: V. Hosu, H. Lin, T. Szirányi, and D. Saupe, “KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment,” *IEEE Transactions on Image Processing*, 2020. Retain the dataset’s original terms of use and attribution when using or redistributing it.

## Random Forest training and evaluation

Install dependencies:

```powershell
python -m pip install -r backend/requirements.txt
cd frontend
npm install
cd ..
```

Train locally (this is separate from application startup):

```powershell
python -m backend.ml.train_model --dataset-root datasets
```

The workflow extracts features from matched image pixels, discards unreadable images with a report, and uses a deterministic 70% train / 15% validation / 15% test split (`random_state=42`). The validation set chooses between two tree counts and `min_samples_leaf` values of 1 or 2; `max_features=1.0`. The selected model is refit on train+validation, and only then scored on the untouched test partition. Metrics are computed from that test partition, not estimated or hardcoded. The joblib artifact stores its version, feature order, preprocessing configuration, target, seed, selected parameters, and test metrics.

Training writes `backend/ml/random_forest_model.joblib` and `backend/ml/training_results.json`. Keep the artifact available to the backend; startup loads it once. If it is missing, health reports `random_forest_ready: false` and analysis returns HTTP 503 with a training instruction. No training occurs on application startup or per upload.

### Measured test-set results

These values are populated from the local training run; no expected or fabricated values are substituted.

| Metric | Untouched test set |
|---|---:|
| MAE | 0.2841 MOS |
| MSE | 0.1349 MOS² |
| RMSE | 0.3673 MOS |
| R² | 0.5644 |
| SRCC | 0.6955 |

These are measured on 1,511 held-out images from the final whole-image extractor run: 10,073 image/MOS pairs matched, 300 local images had no score annotation, no score rows were unmatched, and no matched images were unreadable. The split used 7,051 training, 1,511 validation, and 1,511 test examples. The selected model used 180 trees, `max_features=1.0`, `min_samples_leaf=2`, and seed 42. The metrics are one reproducible experiment, not a guarantee of performance on other data.

An audit trial that changed RF brightness and sharpness to face-region measurements scored MAE 0.3139, RMSE 0.4097, R² 0.4581, and SRCC 0.6307 on the same deterministic split. It was rejected: the whole-image MOS target is more consistently represented by whole-image predictors, while face-aware sharpness stays in the separate CV/FIS context. The accepted version (`rf-koniq10k-1.2.0`) uses the same global extractor for training and inference and recovers the prior baseline metrics; this feature-scope trial did not improve regression performance.

### Training versus local-upload feature distributions

The following are p05 / median / p95 summaries. Training values come from the RF training partition; local-upload values were recomputed with the current inference extractor for 17 unique images in this workspace’s upload history. The upload set is small and self-selected, so these numbers are a comparison probe, not a representative deployment distribution or a formal drift test.

| Feature | KonIQ train p05 / p50 / p95 | Local uploads p05 / p50 / p95 |
|---|---:|---:|
| Brightness | 9.44 / 43.14 / 75.95 | 28.26 / 45.96 / 65.09 |
| Contrast | 12.72 / 45.47 / 79.62 | 34.04 / 54.13 / 72.19 |
| Colorfulness | 3.91 / 32.11 / 59.37 | 0.00 / 23.53 / 38.35 |
| Sharpness | 34.78 / 98.14 / 100.00 | 36.35 / 91.88 / 100.00 |
| Noise | 0.15 / 5.83 / 43.55 | 0.05 / 2.84 / 6.61 |

The clear differences in colorfulness/noise tails, along with the small sample size, warrant larger and better-stratified monitoring before claiming distribution shift or model failure.

Use the full `training_results.json` for dataset-match counts, split sizes, selected parameters, validation scores, and the complete measured metrics. Results are specific to the local dataset files and software versions.

## Mamdani integration and recommendation policy

The existing CV input memberships, legacy rule base, min antecedent conjunction, min implication, max aggregation, centroid defuzzification, and playground are retained. Upload analysis uses a separate Mamdani hybrid rule set so legacy rules cannot conflict with model/MOS rules. Its MOS terms use the actual training MOS distribution (p05≈2.08, median≈3.28, p95≈3.85):

- Low: trapezoid `[1.0, 1.0, 2.1, 2.7]`
- Medium: triangle `[2.4, 3.28, 3.75]`
- High: right shoulder `[3.55, 3.9, 5.0, 5.0]`

These broad overlapping ranges partition the original 1–5 opinion-score scale; they are transparent fuzzy linguistic bands, not fitted action thresholds or KonIQ decision labels. Hybrid rules include direct severe-blur and high-noise defect rules, MOS baselines, sharp-focus corroboration, portrait face evidence, and item object-visibility evidence. Face visibility affects only Portrait Photography; object visibility affects only Item Photography. Landscape composition uses scene saliency and regional sharpness; item framing uses the proposed object bounds. The legacy decision engine’s crisp safety overrides remain for the playground; hybrid recommendations are derived from Mamdani centroid output without a bypass threshold.

Upload hybrid inference receives measured blur, brightness, contrast, noise, composition, and relevant subject evidence alongside RF MOS. Portrait rules consider detected face regions, visibility, and face-weighted exposure; multiple detected faces contribute individually to portrait sharpness. Item rules consider primary-object sharpness, visibility, exposure, and framing, without invoking face detection. Landscape rules consider distributed regional sharpness. These CV rules corroborate the MOS without replacing the trained model or changing KEEP/RECOVER thresholds.

The existing application decision cutoffs are retained: final fuzzy scores **≥70** map to `KEEP`, scores **≥45 and <70** to `RECOVER`, and scores **<45** to `RETAKE`. These are operational thresholds inherited from the SmartSnap decision engine, not labels learned from KonIQ. The 1–5 MOS is also displayed as a linearly normalized `(MOS - 1) × 25` value for readability; the fuzzy engine receives the native MOS value.

### Calibration audit and recommendation evaluation

The audit reproduced the three supplied diagnostic profiles. In the previous upload path, the hybrid FIS received MOS, face visibility, and composition, but not blur; severe blur therefore could not fire a blur rule. Mid-range MOS also activated a moderate baseline without any sharp-focus corroboration, leaving the two clear, sharp examples near the RECOVER boundary. The revised path passes the measured blur explicitly and adds a category-scoped sharp-focus/MOS rule while retaining Mamdani min implication, max aggregation, and centroid defuzzification. The RF training and inference extractor remains global and identical; subject/category-aware blur and exposure are separate CV evidence rather than a different RF input representation.

| Diagnostic probe (not action ground truth) | Before: score / action | After: score / action |
|---|---:|---:|
| Sharp landscape | 54.03 / RECOVER | 70.44 / KEEP |
| Severely blurred street | 53.25 / RECOVER | 20.57 / RETAKE |
| Sharp sports car | 53.25 / RECOVER | 71.03 / KEEP |

The Random Forest MOS estimates for these examples were 3.4176, 2.6383, and 3.3407 respectively. The MOS values did not drive the score change by themselves: the explicit blur rule drove the severe-blur score down, and sharp-focus corroboration moved the two non-portrait cases above the inherited KEEP band.

CV rule context is category-aware: landscapes use scene evidence rather than face detection, portraits use face-region evidence, and item photographs use a proposed object region. Contrast is not treated as universally good or bad; the hybrid FIS does not use it as an isolated positive term.

### Face, fuzzy, and CV metric investigation

The false-positive mountain/reflection image produced one box from `haarcascade_frontalface_default.xml` alone. Other frontal-face cascades did not agree, so the new 2-of-3 consensus rejects it; face count and face sharpness are now zero/absent. In local checks, consensus retained the clearly visible portrait and the intentionally blurred portrait, while excluding a smaller cascade-only proposal in the portrait background. These are targeted local probes, not a general face-detection benchmark.

For the reported legacy FIS input (`blur=18`, `brightness=36`, `contrast=98`, `noise=57`, `face_visibility=75`, `composition=70`), the pre-change result was 68.37 / RECOVER. Membership degrees were: blur Low 1.00; brightness Dark 0.50 and Normal 0.20; contrast High 1.00; noise Medium 0.7083; face Acceptable 0.2273 and Clear 0.4375; composition Average 0.2727 and Good 0.3333. Before calibration, U2 (alpha 0.75 → Good) dominated; R28 (0.40 → Moderate), R18 (0.20 → Excellent), R37 (0.16 → Good), and U1 (0.15 → Good) were also active. U2 rewarded high contrast without considering the Dark exposure membership. The noise membership had no effect: R5 only considers High noise, while R25 also requires Medium blur and Medium contrast, neither of which applied. The face value did not activate any General Photography rule.

The new R39 rule (`noise Medium AND blur Low → Moderate`, weight 0.85) fires at alpha 0.6021 for that profile. Generic positive rich-contrast rules U2, R19, and R33 now also require Normal brightness; the landscape Excellent rule R10 does too. The recalibrated centroid is 63.29 / RECOVER with the original decision cutoffs unchanged. The hybrid path adds H17 (High noise → Poor) and H18 (Medium noise with Low blur → Moderate). With MOS 3.3, blur 10, and composition 70 held constant in a synthetic probe, noise 8 produced 71.37 while noise 100 produced 58.42. These values demonstrate rule sensitivity, not recommendation accuracy.

CV checks use deterministic rock/tree/cloud/reflection texture and local landscape photographs. The synthetic natural scene measured brightness 58.43, contrast 39.82, noise 1.08, and blur 0; applying a 31×31 Gaussian blur raised the blur index to 99. A local mountain/reflection scene measured noise 4.7 and blur 0 after correction. A flat synthetic frame with high independent sensor noise had noise 100 and blur 96.31 rather than false sharpness. The contrast measure remains a global tonal-spread statistic, not a goodness score; e.g., a value near 98 is high contrast and only becomes meaningful through rule context. These representative probes do not cover every camera pipeline or texture.

The supplied examples and a small set of local uploads were visually inspected as qualitative probes spanning landscapes, portraits, multi-face portraits, and general photography. They are regression probes, not independently adjudicated action labels: supplied expected outcomes are not ground truth, MOS is not an action label, and no KEEP/RECOVER/RETAKE classification metrics are reported. This workspace has no independently labeled category-specific diagnostic corpus, including reviewed item, noise, low-contrast, and exposure challenge sets; the recommendation policy is not scientifically calibrated as a classifier.

For a valid action-assessment benchmark, collect Portrait, Item, Landscape, and General photographs with sharp, average, exposure-defective, severely blurred, noisy, and low-contrast examples. Include subject/background focus differences, varied object types and clutter, partially obscured subjects, and different face counts. Have multiple reviewers independently and blindly label intended-use utility (`KEEP`: usable as captured; `RECOVER`: useful after plausible edits; `RETAKE`: a material defect prevents the intended use), then adjudicate disagreements. Keep scenes/photographers separate across calibration and held-out evaluation. Report reviewer agreement and per-class precision/recall/F1, macro-F1, and confusion matrices only against the adjudicated labels. Do not derive action labels from MOS, tune on the supplied cases, or treat a unit-test expectation as a benchmark result.

The revised unit tests are limited regression checks of rule activation, direction of score change, category invariance, and the severe-blur decision band. They do not measure recommendation accuracy or establish generalization.

## Run the application

Build the React application and run the API:

```powershell
cd frontend
npm run build
cd ..
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. For frontend hot reload, run `python -m uvicorn backend.app:app --reload` and `npm run dev` in separate terminals from the project root and `frontend`, respectively.

`POST /api/analyze` retains the upload contract and existing CV diagnostics, and adds a `random_forest` object (`predicted_mos`, normalized display score, model version, and input features). `fuzzy` contains the final centroid score and activated rules; `decision` contains the final recommendation, issues, and improvement suggestions. `GET /api/health` reports model readiness and version.

## Tests

```powershell
python -m pytest backend/tests/ -v
cd frontend
npm run build
```

Hybrid tests cover image/annotation matching, unmatched and invalid data, feature consistency, training and loading, MOS prediction, fuzzy integration, recommendations, API response shape, and invalid uploads. Existing tests cover the original CV and fuzzy behavior.

## Limitations and future work

- The Random Forest is a handcrafted-feature baseline; it may not capture semantic preferences, subject importance, or all photographic genres.
- KonIQ MOS reflects the dataset’s annotators and image distribution; performance on other cameras, domains, and populations may differ.
- Face detection and item foreground proposals are heuristic CV measures; missed/occluded faces, off-center objects, and clutter can reduce reliability. An uncertain item proposal remains neutral.
- `KEEP`/`RECOVER`/`RETAKE` thresholds and fuzzy bands are documented application policy, not ground-truth labels validated against retake outcomes.
- The diagnostic probes are small, local, and not independently action-labeled; no evidence currently supports classification-performance claims or complete category-specific calibration.
- Future work could compare cross-validated model families, calibrate fuzzy policy with a separately labeled action dataset, and test generalization on an external IQA benchmark.

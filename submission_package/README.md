# Traffic Demand Forecasting — v3

## Problem
Predict `demand` (traffic volume) at specific (day, minute, geohash) locations. Training data has 24 days (0-23) × 4 minutes (0/15/30/45). Test asks for predictions on days 2-13.

## Key Insight: RoadType Bounds
Training data reveals **exact demand boundaries** per RoadType:
- **Residential**: [0.00, 0.22] — demand is near-deterministic within this band
- **Street**: [0.22, 0.35] — narrow band, no overlap
- **Highway**: [0.35, 1.00] — wider band, higher variance

These are strict — not a single training sample violates them. Post-clipping to these bounds enforces domain knowledge.

## Approach

### Cross-Validation
**Forward-chaining time-based CV** (train on early days, validate on later days) instead of random KFold. This gives honest temporal generalization estimates.

### Feature Engineering
| Feature | Description |
|---|---|
| **K-fold target encoding** | `(RoadType, geohash_6)` smoothed mean demand — dominant signal (54% importance) |
| **Additional encodings** | `geohash_6`, `(RoadType, minute)`, `(RoadType, day)` — all K-fold |
| **Temporal** | day (numeric+poly+sin/cos), minute (categorical+sin/cos) |
| **Spatial** | lat/lng decoded, geohash frequency, highway ratio per region |
| **Interactions** | road_tier × day, road_tier × minute, road_tier × large_vehicles |
| **Missing** | Binary flags for RoadType, Temperature, Weather |

### Modeling
- **CatBoost** — 3000 iterations, depth 8, subsample 0.8, 3 seeds
- **LightGBM** — max_depth 8, 63 leaves, subsample 0.8, 3 seeds
- **XGBoost** — max_depth 7, subsample 0.8, 3 seeds
- **Weighted Ensemble** — weights computed from time-based OOF R²

### Post-Processing
1. **Highway calibration**: Per-day scaling to match training Highway mean (corrects systematic underprediction from over-regularization)
2. **RoadType bounds**: Predictions clipped to known demand ranges per RoadType

## Results (Time-Based CV)

| Model | CV R² |
|---|---|
| CatBoost (seed=42) | 0.804 ± 0.066 |
| LightGBM (seed=42) | 0.811 ± 0.064 |
| XGBoost (seed=42) | 0.802 ± 0.065 |
| **Ensemble** | **est. 0.80-0.83 (time-CV)** |

Fold 1 (val days 9-13, matching test range) achieves **0.93-0.94 R²**, suggesting strong performance on test domain.

## Dependencies
- Python 3.9+
- pandas, numpy
- scikit-learn (KFold, LabelEncoder, r2_score)
- catboost ≥ 1.2
- lightgbm ≥ 4.0
- xgboost ≥ 2.0
- geohash2

## Files
| File | Description |
|---|---|
| `solution.py` | Full pipeline script |
| `solution.ipynb` | Jupyter notebook |
| `submission.csv` | Predictions (41,778 rows) |
| `requirements.txt` | Dependencies |

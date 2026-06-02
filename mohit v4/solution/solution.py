#!/usr/bin/env python3
"""
Traffic Demand Forecasting v3 — K-fold target encoding, time-based CV, ensemble.
Metric: R² Score
"""
import os, sys, warnings, gc, math, random
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

BASE = Path(__file__).parent.resolve()

# ── 1. Load ──────────────────────────────────────────────────────────────
train = pd.read_csv(BASE / 'train.csv')
test  = pd.read_csv(BASE / 'test.csv')
print(f"Train: {train.shape}  Test: {test.shape}")

# ── 2. Parse timestamps (format: day:minute) ─────────────────────────────
def parse_time(df):
    df = df.copy()
    parts = df['timestamp'].str.split(':', expand=True)
    df['day'] = parts[0].astype(int)
    df['minute'] = parts[1].astype(int)
    df['slot'] = df['minute'] // 15
    return df

train = parse_time(train)
test  = parse_time(test)

print(f"Train days: {sorted(train['day'].unique())}")
print(f"Test  days: {sorted(test['day'].unique())}")
print(f"Train minutes: {sorted(train['minute'].unique())}")
print(f"Test  minutes: {sorted(test['minute'].unique())}")

# ── 3. K-fold target encoding ───────────────────────────────────────────
def kfold_target_encode(df_train, df_test, group_cols, target='demand',
                         n_folds=10, smooth=10):
    """
    Proper K-fold target encoding with smoothing.
    Returns encoded series for train (K-fold) and test (full-data).
    """
    prior = df_train[target].mean()
    train_enc = np.zeros(len(df_train))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(df_train):
        fold_tr = df_train.iloc[tr_idx]
        fold_va = df_train.iloc[va_idx]

        gb = fold_tr.groupby(group_cols)[target]
        means = gb.mean()
        counts = gb.count()
        smoothed = (counts * means + prior * smooth) / (counts + smooth)

        # Map to validation
        key = fold_va.set_index(group_cols).index
        vals = key.map(smoothed)
        train_enc[va_idx] = vals.values
        # Fallback: group mean within first group column
        if vals.isna().any():
            nan_idx = np.where(pd.isna(vals))[0]
            for i in nan_idx:
                orig_idx = va_idx[i]
                row = df_train.iloc[orig_idx]
                fallback = fold_tr.groupby(group_cols[0])[target].mean()
                train_enc[orig_idx] = fallback.get(row[group_cols[0]], prior)

    # For test: use full training data
    gb_full = df_train.groupby(group_cols)[target]
    means_full = gb_full.mean()
    counts_full = gb_full.count()
    smoothed_full = (counts_full * means_full + prior * smooth) / (counts_full + smooth)
    test_key = df_test.set_index(group_cols).index
    test_vals = test_key.map(smoothed_full)
    test_enc = test_vals.values.copy()
    # Fallback
    if pd.isna(test_enc).any():
        nan_mask = pd.isna(test_enc)
        fallback = df_train.groupby(group_cols[0])[target].mean()
        for i in np.where(nan_mask)[0]:
            test_enc[i] = fallback.get(df_test.iloc[i][group_cols[0]], prior)

    return train_enc, test_enc

# ── 4. Feature Engineering ──────────────────────────────────────────────
def engineer(df, is_train=False, train_full=None):
    df = df.copy()

    df['geo_4'] = df['geohash'].str[:4]
    df['geo_5'] = df['geohash'].str[:5]
    df['geo_6'] = df['geohash'].str[:6]

    # Temporal
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 24)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 24)
    df['day_sq'] = df['day'] ** 2
    df['day_cub'] = df['day'] ** 3
    df['min_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['min_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['slot_sin'] = np.sin(2 * np.pi * df['slot'] / 4)
    df['slot_cos'] = np.cos(2 * np.pi * df['slot'] / 4)

    # Geohash → lat/lng
    try:
        import geohash2
        df['lat'], df['lng'] = zip(*df['geohash'].apply(
            lambda g: geohash2.decode(g) if pd.notna(g) else (np.nan, np.nan)))
    except Exception:
        df['lat'] = df['geohash'].apply(lambda g: sum(ord(c) for c in str(g)[:3]) if pd.notna(g) else 0)
        df['lng'] = df['geohash'].apply(lambda g: sum(ord(c) for c in str(g)[3:6]) if pd.notna(g) else 0)

    # Missing flags
    for c in ['RoadType', 'Temperature', 'Weather']:
        df[f'{c}_missing'] = df[c].isna().astype(int)

    # RoadType
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    rt_map = {'Residential': 0, 'Street': 1, 'Highway': 2, 'Unknown': -1}
    df['road_tier'] = df['RoadType'].map(rt_map)
    df['is_highway'] = (df['RoadType'] == 'Highway').astype(int)
    df['is_street']  = (df['RoadType'] == 'Street').astype(int)
    df['is_residential'] = (df['RoadType'] == 'Residential').astype(int)

    # Numerical
    df['NumberofLanes'] = df['NumberofLanes'].fillna(-1).astype(int)
    df['has_many_lanes'] = (df['NumberofLanes'] >= 4).astype(int)
    lv_map = {'Not Allowed': 0, 'Allowed': 1}
    df['LargeVehicles_num'] = df['LargeVehicles'].map(lv_map).fillna(-1)
    df['Landmarks_num'] = df['Landmarks'].map({'No': 0, 'Yes': 1}).fillna(-1)

    # Weather
    df['Weather'] = df['Weather'].fillna('Unknown')
    w_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3, 'Unknown': -1}
    df['weather_code'] = df['Weather'].map(w_map)

    # Temperature imputation
    temp_prior = df['Temperature'].median()
    df['Temperature'] = df.groupby(['day', 'geo_5'])['Temperature'].transform(
        lambda s: s.fillna(s.median())).fillna(temp_prior)

    # Composite identifiers
    df['rt_lanes_lv'] = (
        df['RoadType'].astype(str) + '_' +
        df['NumberofLanes'].astype(str) + '_' +
        df['LargeVehicles'].astype(str)
    )

    # Interactions
    df['tier_x_day']   = df['road_tier'] * df['day']
    df['tier_x_minute'] = df['road_tier'] * df['minute']
    df['tier_x_lv']    = df['road_tier'] * df['LargeVehicles_num']
    df['day_x_minute'] = df['day'] * df['minute']

    # Geohash frequency (from full dataset)
    geo6_counts = df['geo_6'].value_counts().to_dict()
    df['geo_6_freq'] = df['geo_6'].map(geo6_counts) / len(df)
    geo5_counts = df['geo_5'].value_counts().to_dict()
    df['geo_5_freq'] = df['geo_5'].map(geo5_counts) / len(df)

    # Highway ratio within geo_6 region
    geo6_rt = df.groupby(['geo_6', 'is_highway']).size().unstack(fill_value=0)
    geo6_hwy_ratio = geo6_rt[1] / (geo6_rt[0] + geo6_rt[1])
    df['geo_6_highway_ratio'] = df['geo_6'].map(geo6_hwy_ratio).fillna(0)

    return df

# Apply engineering
train_fe = engineer(train, is_train=True)
test_fe  = engineer(test, is_train=False)

# ── 5. K-fold Target Encoding features ──────────────────────────────────
print("\n=== Computing target encodings... ===")

enc_rt_geo6_tr, enc_rt_geo6_te = kfold_target_encode(
    train_fe, test_fe, ['RoadType', 'geo_6'], smooth=5)
train_fe['enc_rt_geo6'] = enc_rt_geo6_tr
test_fe['enc_rt_geo6']  = enc_rt_geo6_te

enc_geo6_tr, enc_geo6_te = kfold_target_encode(
    train_fe, test_fe, ['geo_6'], smooth=30)
train_fe['enc_geo6'] = enc_geo6_tr
test_fe['enc_geo6']  = enc_geo6_te

enc_rt_min_tr, enc_rt_min_te = kfold_target_encode(
    train_fe, test_fe, ['RoadType', 'minute'], smooth=30)
train_fe['enc_rt_minute'] = enc_rt_min_tr
test_fe['enc_rt_minute']  = enc_rt_min_te

enc_rt_day_tr, enc_rt_day_te = kfold_target_encode(
    train_fe, test_fe, ['RoadType', 'day'], smooth=30)
train_fe['enc_rt_day'] = enc_rt_day_tr
test_fe['enc_rt_day']  = enc_rt_day_te

print("  enc_rt_geo6  : Done")
print("  enc_geo6     : Done")
print("  enc_rt_minute: Done")
print("  enc_rt_day   : Done")

# ── 6. Define feature sets ──────────────────────────────────────────────
CAT_FEATURES = ['RoadType', 'Weather', 'geo_4', 'geo_5', 'geo_6',
                'rt_lanes_lv', 'LargeVehicles', 'Landmarks']

NUM_FEATURES = [
    # Temporal
    'day', 'minute', 'slot', 'day_sin', 'day_cos', 'day_sq', 'day_cub',
    'min_sin', 'min_cos', 'slot_sin', 'slot_cos',
    # Road
    'road_tier', 'is_highway', 'is_street', 'is_residential',
    # Numerical
    'NumberofLanes', 'has_many_lanes', 'LargeVehicles_num', 'Landmarks_num',
    'weather_code', 'Temperature',
    # Missing
    'RoadType_missing', 'Temperature_missing', 'Weather_missing',
    # Interactions
    'tier_x_day', 'tier_x_minute', 'tier_x_lv', 'day_x_minute',
    # Spatial
    'lat', 'lng', 'geo_6_freq', 'geo_5_freq', 'geo_6_highway_ratio',
    # Target encodings
    'enc_rt_geo6', 'enc_geo6', 'enc_rt_minute', 'enc_rt_day',
]

NUM_FEATURES = [c for c in NUM_FEATURES if c in train_fe.columns]
CAT_FEATURES = [c for c in CAT_FEATURES if c in train_fe.columns]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
print(f"\nFeatures: {len(ALL_FEATURES)} ({len(NUM_FEATURES)} num + {len(CAT_FEATURES)} cat)")

# ── 7. Prepare matrices ──────────────────────────────────────────────────
# Label encode categoricals for LGB/XGB
from sklearn.preprocessing import LabelEncoder
le_dict = {}
for col in CAT_FEATURES:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]]).astype(str).fillna('NAN')
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].astype(str).fillna('NAN'))
    test_fe[col]  = le.transform(test_fe[col].astype(str).fillna('NAN'))
    le_dict[col] = le

X_train = train_fe[ALL_FEATURES].values.astype(np.float32)
y_train = train_fe['demand'].values.astype(np.float32)
X_test  = test_fe[ALL_FEATURES].values.astype(np.float32)

# CatBoost DataFrames (categoricals as strings)
X_train_cb = pd.DataFrame(X_train, columns=ALL_FEATURES)
X_test_cb  = pd.DataFrame(X_test, columns=ALL_FEATURES)
for col in CAT_FEATURES:
    X_train_cb[col] = X_train_cb[col].astype(int).astype(str)
    X_test_cb[col]  = X_test_cb[col].astype(int).astype(str)

# ── 8. Time-based Cross-Validation ──────────────────────────────────────
# Forward chaining by day
train_days = sorted(train_fe['day'].unique())
n_days = len(train_days)

time_folds = []
fold_defs = [
    (int(n_days * 0/5), int(n_days * 2/5), int(n_days * 3/5)),
    (int(n_days * 0/5), int(n_days * 3/5), int(n_days * 4/5)),
    (int(n_days * 0/5), int(n_days * 4/5), n_days),
    (int(n_days * 1/5), int(n_days * 4/5), n_days),
    (int(n_days * 2/5), int(n_days * 4/5), n_days),
]

for start, mid, end in fold_defs:
    tr_days = train_days[start:mid]
    va_days = train_days[mid:end]
    if len(va_days) == 0:
        continue
    tr_idx = train_fe.index[train_fe['day'].isin(tr_days)]
    va_idx = train_fe.index[train_fe['day'].isin(va_days)]
    time_folds.append((tr_idx.values, va_idx.values))

print(f"\n=== Time-based CV: {len(time_folds)} folds ===")
for i, (tr_idx, va_idx) in enumerate(time_folds):
    tr_days = sorted(train_fe.loc[tr_idx, 'day'].unique())
    va_days = sorted(train_fe.loc[va_idx, 'day'].unique())
    print(f"  Fold {i+1}: train days {tr_days[0]}-{tr_days[-1]} ({len(tr_idx)} rows) "
          f"→ val days {va_days[0]}-{va_days[-1]} ({len(va_idx)} rows)")

# ── 9. Model training & CV ──────────────────────────────────────────────
def train_fold(model_type, params, tr_idx, va_idx, seed=42):
    np.random.seed(seed * 10)
    params = params.copy()

    if model_type == 'catboost':
        params['random_seed'] = seed
        X_tr = X_train_cb.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train_cb.iloc[va_idx]
        y_va = y_train[va_idx]

        m = CatBoostRegressor(**params).fit(
            X_tr, y_tr, eval_set=(X_va, y_va),
            cat_features=CAT_FEATURES, verbose=False,
            early_stopping_rounds=100)
        pred = m.predict(X_va)

    elif model_type == 'lgb':
        params['random_state'] = seed
        X_tr = X_train[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train[va_idx]
        y_va = y_train[va_idx]
        params['n_estimators'] = 10000

        m = lgb.LGBMRegressor(**params).fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        pred = m.predict(X_va, num_iteration=m.best_iteration_)

    else:  # xgb
        params['random_state'] = seed
        X_tr = X_train[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train[va_idx]
        y_va = y_train[va_idx]
        params['n_estimators'] = 10000

        m = xgb.XGBRegressor(**params).fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = m.predict(X_va)

    score = r2_score(y_va, pred)
    return score, m, pred

def cv_scores(model_type, params, seeds=[42]):
    all_scores = []
    all_models = []
    oof_preds = np.zeros(len(train_fe))
    oof_counts = np.zeros(len(train_fe))

    for seed in seeds:
        fold_scores = []
        fold_models = []
        for i, (tr_idx, va_idx) in enumerate(time_folds):
            score, m, pred = train_fold(model_type, params, tr_idx, va_idx, seed)
            fold_scores.append(score)
            fold_models.append(m)
            # Accumulate OOF predictions
            oof_preds[va_idx] += pred
            oof_counts[va_idx] += 1
            print(f"  {model_type} seed={seed} Fold {i+1}: R² = {score:.6f}")
        mean_s = np.mean(fold_scores)
        std_s  = np.std(fold_scores)
        all_scores.append((fold_scores, mean_s, std_s))
        all_models.append(fold_models)
        print(f"  → {model_type} seed={seed}: CV R² = {mean_s:.6f} ± {std_s:.6f}")

    oof_preds = np.divide(oof_preds, oof_counts, out=np.zeros_like(oof_preds), where=oof_counts > 0)
    return all_scores, all_models, oof_preds

# ── Define model params ─────────────────────────────────────────────────
CATBOOST_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.03, 'depth': 8,
    'l2_leaf_reg': 5, 'random_strength': 1.0,
    'loss_function': 'RMSE', 'eval_metric': 'R2',
    'bootstrap_type': 'Bernoulli', 'subsample': 0.8,
    'max_ctr_complexity': 4, 'min_data_in_leaf': 10,
    'verbose': 0, 'od_type': 'Iter', 'od_wait': 100,
}

LGB_PARAMS = {
    'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 1.0, 'reg_lambda': 1.0,
    'min_child_samples': 30, 'min_child_weight': 10,
    'verbose': -1, 'metric': 'rmse', 'boosting_type': 'gbdt',
}

XGB_PARAMS = {
    'learning_rate': 0.03, 'max_depth': 7,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 1.0, 'reg_lambda': 1.0, 'min_child_weight': 10,
    'gamma': 0.1, 'verbosity': 0,
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist',
}

print("\n" + "=" * 60)
print("TRAINING MODELS WITH TIME-BASED CV")
print("=" * 60)

all_oof = {}
all_results = {}

for name, mtype, params, seeds in [
    ('CatBoost', 'catboost', CATBOOST_PARAMS, [42, 123, 456]),
    ('LightGBM', 'lgb', LGB_PARAMS, [42, 123, 456]),
    ('XGBoost', 'xgb', XGB_PARAMS, [42, 123, 456]),
]:
    print(f"\n--- {name} ---")
    scores, models, oof = cv_scores(mtype, params, seeds)
    all_oof[name] = oof
    all_results[name] = (scores, models)
    gc.collect()

# ── 10. Ensemble ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ENSEMBLE")
print("=" * 60)

# Compute ensemble weights from OOF R²
weights = {}
total_r2 = 0
for name in all_oof:
    r2 = r2_score(y_train, all_oof[name])
    weights[name] = max(r2, 0)  # clip negative to 0
    total_r2 += weights[name]
    print(f"  {name} OOF R² = {r2:.6f}")

for name in weights:
    weights[name] /= total_r2 if total_r2 > 0 else len(weights)
    print(f"  {name} ensemble weight = {weights[name]:.4f}")

# Blend OOF predictions
blend_oof = np.zeros(len(train_fe))
for name in all_oof:
    blend_oof += weights[name] * all_oof[name]
blend_r2 = r2_score(y_train, blend_oof)
print(f"\n  Blended OOF R² = {blend_r2:.6f}")

# ── 11. Generate test predictions from fold models ──────────────────────
print("\n" + "=" * 60)
print("PREDICTING TEST WITH FOLD MODELS (no full retrain)")
print("=" * 60)

model_avgs = {}

for name, mtype, seeds in [
    ('CatBoost', 'catboost', [42, 123, 456]),
    ('LightGBM', 'lgb', [42, 123, 456]),
    ('XGBoost', 'xgb', [42, 123, 456]),
]:
    _, fold_models = all_results[name]
    per_model_preds = []
    for seed_idx, seed in enumerate(seeds):
        for fm in fold_models[seed_idx]:
            if mtype == 'catboost':
                pred = fm.predict(X_test_cb)
            elif mtype == 'lgb':
                pred = fm.predict(X_test, num_iteration=fm.best_iteration_)
            else:
                pred = fm.predict(X_test)
            per_model_preds.append(pred)
        print(f"  {name} seed={seed}: {len(fold_models[seed_idx])} fold models done")
    per_model_preds = np.clip(np.array(per_model_preds), 0.0, 1.0)
    model_avgs[name] = np.mean(per_model_preds, axis=0)
    print(f"  {name} avg range: [{model_avgs[name].min():.4f}, {model_avgs[name].max():.4f}]")

final_pred = sum(weights[n] * model_avgs[n] for n in model_avgs)
print(f"\n  Ensemble range: [{final_pred.min():.6f}, {final_pred.max():.6f}]")

# ── 12. Post-hoc Highway calibration ─────────────────────────────────────
# Adjust per-day Highway mean to match training per-day mean
test_roadtype = test['RoadType'].fillna('Unknown')
hwy_train_mean = train[train['RoadType'] == 'Highway'].groupby('day')['demand'].mean()
hwy_mask = test_roadtype == 'Highway'
n_calibrated = 0
for d in sorted(test_fe['day'].unique()):
    day_hwy_mask = hwy_mask & (test_fe['day'] == d)
    if day_hwy_mask.sum() > 0 and d in hwy_train_mean.index:
        train_day_mean = hwy_train_mean[d]
        pred_day_mean = final_pred[day_hwy_mask].mean()
        if pred_day_mean > 0:
            scale = train_day_mean / pred_day_mean
            final_pred[day_hwy_mask] = final_pred[day_hwy_mask] * scale
            n_calibrated += day_hwy_mask.sum()

print(f"\n  Highway calibration: {n_calibrated} samples adjusted")
print(f"  Per-day scaling factors:")
for d in sorted(test_fe['day'].unique()):
    day_hwy_mask = hwy_mask & (test_fe['day'] == d)
    if day_hwy_mask.sum() > 0 and d in hwy_train_mean.index:
        print(f"    Day {d}: train_mean={hwy_train_mean[d]:.4f}  "
              f"pred_mean_after={final_pred[day_hwy_mask].mean():.4f}")
print(f"  After calibration range: [{final_pred.min():.6f}, {final_pred.max():.6f}]")

# ── 13. Post-processing: RoadType bounds ─────────────────────────────────
print("\n  Applying RoadType demand bounds...")

test_roadtype = test['RoadType'].fillna('Unknown')
mask_res = test_roadtype == 'Residential'
mask_st  = test_roadtype == 'Street'
mask_hwy = test_roadtype == 'Highway'
mask_unk = ~(mask_res | mask_st | mask_hwy)

before = final_pred.copy()

final_pred = np.clip(final_pred, 0.0, 1.0)
final_pred = np.where(mask_res, np.clip(final_pred, 0.0, 0.22), final_pred)
final_pred = np.where(mask_st,  np.clip(final_pred, 0.22, 0.35), final_pred)
final_pred = np.where(mask_hwy, np.clip(final_pred, 0.35, 1.0), final_pred)

n_clipped = np.sum(final_pred != before)
print(f"  Clipped {n_clipped}/{len(final_pred)} predictions ({100*n_clipped/len(final_pred):.1f}%)")
for label, m in [('Residential', mask_res), ('Street', mask_st), ('Highway', mask_hwy), ('Unknown', mask_unk)]:
    if m.sum() > 0:
        vals = final_pred[m]
        print(f"    {label}: mean={vals.mean():.4f} min={vals.min():.4f} max={vals.max():.4f} n={m.sum()}")
print(f"  Final range: [{final_pred.min():.6f}, {final_pred.max():.6f}]")

# ── 14. Save submission ──────────────────────────────────────────────────
sub = pd.DataFrame({'Index': test['Index'].values, 'demand': final_pred})
sub.to_csv(BASE / 'submission.csv', index=False)
print(f"\n  Saved submission.csv ({len(sub)} rows)")

assert list(sub.columns) == ['Index', 'demand']
assert len(sub) == len(test)
print("  Submission format verified ✓")

# ── 15. Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Train: {train.shape[0]} rows, {len(ALL_FEATURES)} features")
print(f"  Test:  {test.shape[0]} rows")
print(f"  Time-based CV folds: {len(time_folds)}")
print(f"  Models: CatBoost, LightGBM, XGBoost × 3 seeds each")
print(f"  Ensemble OOF R²: {blend_r2:.6f}")
print(f"  RoadType bounds applied: Yes")
print(f"  Submission: submission.csv")
print("Done!")

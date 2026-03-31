# Supply Chain  (Late Delivery Risk Prediction)

Predicting whether an order will be delivered late using the DataCo Smart Supply Chain dataset.

---

## Dataset
- 180,519 orders
- 53 raw features
- Target: `Late_delivery_risk` (1 = Late, 0 = On Time)
- Class balance: 54.8% Late / 45.2% On Time

---

## Problem Statement

A supply chain company loses money from late deliveries - unhappy customers, not meeting the promised service standard , and operational cost. The goal is to predict late deliveries **before they happen** so the business can anticipate and act before issues arise.

---

## Project Structure

```
supply_chain/
├── datasets/
│   └── DataCoSupplyChainDataset.csv
├── models/
│   └── pipeline.pkl
├── supply_chain_final.ipynb
└── README.md
```

---

## Approach

### Data Cleaning
- Dropped 100% null columns, PII, leakage columns, random IDs, and mathematical duplicates
- Removed post-delivery features (`Delivery Status`, `Days for shipping (real)`, `Order Status`) to prevent leakage
- Cleaned non-ASCII characters in text columns

### Feature Selection
- **Geography dropped** — Region std=0.017, only 9 percentage point spread across entire world. Geography does not predict late delivery in this dataset
- **High-cardinality columns dropped** - `Order City` (3,594 unique), `Order State` (1,089 unique) would cause memorization not learning(Overfit)
- **Zero-variance column dropped** - `Product Status` (all 0s)
- **17 clean features** kept with documented business reasoning

### Preprocessing
- Train/test split before any encoding (80/20, stratified)
- One-Hot Encoding for low cardinality categoricals (≤11 unique)
- Ordinal Encoding for medium-high cardinality (`Category Name`, `Product Name`)
- StandardScaler applied only for Logistic Regression and KNN — tree models don't need scaling
- All transformers fit on train only, applied to test — no leakage

### Model Comparison

| Model | AUC | Recall | FN | Overfit Gap |
|---|---|---|---|---|
| Gradient Boosting | 0.7431 | 0.546 | 8,988 | 0.006 ✅ |
| Logistic Regression | 0.7421 | 0.547 | 8,981 | 0.000 ✅ |
| **XGBoost** | **0.7390** | **0.574** | **8,433** | **0.066 ⚠️** |
| Random Forest | 0.7316 | 0.633 | 7,264 | 0.268 ❌ |
| KNN | 0.7078 | 0.659 | 6,750 | — |

**XGBoost chosen** — best F1, best recall among non-overfitted models, fastest to tune.

### Tuning
- `RandomizedSearchCV` with 50 iterations, 5-fold CV, `scoring='f1'`
- Early stopping revealed `learning_rate=0.2, n_estimators=500` from grid search was overfitting — test AUC declined from tree 0
- Fixed with `learning_rate=0.05` + `early_stopping_rounds=30` → optimal at iteration 57
- Overfit gap reduced from 0.066 → 0.002

### Threshold Tuning
Default threshold (0.5) is too conservative — misses too many late deliveries. In supply chain, missing a late delivery (customer impact, not meeting service standard) is more costly than a false alarm (minor operational intervention).

| Threshold | FN (Missed Late) | FP (False Alarms) | F1 |
|---|---|---|---|
| 0.41 | 1,971 | 12,441 | **0.712** |
| 0.45 | 8,263 | 2,753 | 0.677 |
| 0.50 | 8,830 | 2,121 | 0.667 |

Threshold 0.41 chosen - best F1, catches the most late deliveries.

---

## Final Results

| Metric | Value |
|---|---|
| CV AUC | **0.7426 ± 0.0032** |
| Test AUC | 0.7452 |
| Accuracy | 0.67 |
| Precision (Late) | 0.72 |
| Recall (Late) | 0.67 |
| F1 (Late) | 0.71 |
| Missed Late (FN) | 1,971 |
| False Alarms (FP) | 12,441 |
| Overfit Gap | 0.002 |

CV std of ±0.0032 confirms the model is stable and not getting lucky on a single split.

---

## Key Findings

### 1. Shipping Mode drives everything
Feature importance shows Shipping Mode accounts for 79% of model importance - confirmed by EDA.

| Shipping Mode | Late Rate | Model Importance |
|---|---|---|
| Standard Class | 38% | 62% |
| First Class | 95% | 8% |
| Same Day | 46% | 7% |
| Second Class | 77% | 2% |

First Class has a 95% late delivery rate despite being a premium service.

### 2. Geography is irrelevant
Region std = 0.017, only a 9 percentage point spread across all world regions. Where the order is going does not predict late delivery in this dataset.

### 3. Threshold tuning matters
Moving threshold 0.5 → 0.41 reduced missed late deliveries from 8,908 → 1,971.

### 4. Random Forest severely overfit
Train AUC = 1.000, Test AUC = 0.732, Gap = 0.268. Default RF grows trees to full depth and memorizes training data entirely.

---

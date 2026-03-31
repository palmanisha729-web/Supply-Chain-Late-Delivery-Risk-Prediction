# Supply Chain  (Late Delivery Risk Prediction)

Predicting whether an order will be delivered late using the DataCo Smart Supply Chain dataset.

---

## Dataset

[DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

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

Install all at once:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

---
## Usage

```python
import joblib
import pandas as pd

pipeline = joblib.load('models/pipeline.pkl')

new_order = pd.DataFrame([{
    'Type': 'DEBIT',
    'Days for shipment (scheduled)': 4,
    'Category Name': 'Fishing',
    'Customer Segment': 'Consumer',
    'Department Name': 'Outdoors',
    'Order Item Discount': 10.0,
    'Order Item Discount Rate': 0.05,
    'Order Item Product Price': 120.0,
    'Order Item Profit Ratio': 0.25,
    'Order Item Quantity': 2,
    'Sales': 240.0,
    'Order Profit Per Order': 30.0,
    'Product Name': 'Fishing Rod',
    'Shipping Mode': 'First Class',
    'order_month': 3,
    'order_dayofweek': 2,
    'order_quarter': 1
}])

proba = pipeline.predict_proba(new_order)[:, 1][0]
pred  = int(proba >= 0.41)

print(f"Late delivery probability: {proba:.2%}")
print(f"Prediction: {'Late' if pred == 1 else 'On Time'}")
```

---

## Limitations & Future Work

**Why AUC is capped at ~0.74**

The available features are order-level (what was ordered, how it was shipped). Late delivery is fundamentally an operational/logistics problem. Features that would meaningfully improve the model:

- [Carrier-level historical on-time rates](https://www.bts.gov)
- [Warehouse-to-customer distance](https://geopy.readthedocs.io)
- [Warehouse stock levels at order time](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- [Carrier capacity utilization](https://www.bts.gov)
- [Weather / seasonal disruption data](https://open-meteo.com)

---

## References

Citations for tools, libraries, and data directly used in this project.

**Dataset**
> Constante, F., Silva, F., & Pereira, A. (2019). *DataCo Smart Supply Chain 
> for Big Data Analysis* (Version 5). Mendeley Data.
> https://doi.org/10.17632/8gx2fvg2k6.5

**Algorithms & Libraries**
> Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
> *Proceedings of the 22nd ACM SIGKDD*, 785–794.
> https://doi.org/10.1145/2939672.2939785

> Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
> *Journal of Machine Learning Research*, 12, 2825–2830.
> https://jmlr.org/papers/v12/pedregosa11a.html

**Concepts & Documentation**
- [IBM — What is PII](https://www.ibm.com/think/topics/pii)
- [Matplotlib bar_label](https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.html)
- [Matplotlib spines](https://matplotlib.org/stable/gallery/spines/spines.html)
- [Scikit-learn — Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [XGBoost Early Stopping](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
- [Threshold-Moving for Imbalanced Classification](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification)
- [Featuretools — Time Series Features](https://docs.featuretools.com/en/stable/guides/time_series.html)
- [ScienceDirect — Temporal Feature](https://www.sciencedirect.com/topics/computer-science/temporal-feature)

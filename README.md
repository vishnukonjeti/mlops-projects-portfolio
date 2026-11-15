# MLOps Projects Portfolio

A collection of end-to-end MLOps projects covering data validation, experiment tracking, hyperparameter tuning, drift monitoring and model explainability.

## Projects

| # | Project | Status |
|---|---------|--------|
| 1 | [Predictive Maintenance Classification](project%201/) | Complete |
| 2 | Project 2 (`project 2/`) | Scaffold only (empty folders) |

### Project 1: Predictive Maintenance Classification

Multi-class failure type prediction for industrial machines (Industrial IoT / Manufacturing). The failure classes are TWF, HDF, PWF and OSF. The data is heavily imbalanced, with about 97% of records showing no failure.

**Pipeline** (notebook: `project 1/MLOps_Assignment_PredMaint_Starter.ipynb`)

1. **Data validation and EDA**: Pandera schema checks, class distribution analysis, and engineered features (mechanical power `Power_W`, temperature differential `Temp_diff`).
2. **Experiment tracking and model selection**: SMOTE applied to the training split only. Logistic Regression, Random Forest, XGBoost and LightGBM are compared in MLflow.
3. **Tuning and registry**: Optuna (30 trials, TPE sampler) tunes XGBoost. The best model is registered in the MLflow Model Registry.
4. **Drift detection**: Evidently reports on the `current` and `stress` batches, saved as `drift_current.html` and `drift_stress.html`.
5. **Explainability**: Per-class SHAP analysis, saved as `shap_per_class.png`.

**Headline result:** tuned XGBoost reached a Macro F1 of about **0.761**, up from 0.748 for the baseline.

**Tools:** pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, MLflow, Optuna, Evidently, SHAP, Pandera.

**Layout**

```
project 1/
├── MLOps_Assignment_PredMaint_Starter.ipynb
├── data/               # train / current / stress / raw CSVs
├── mlruns/             # MLflow runs and model artifacts
├── mlflow.db           # MLflow tracking store
├── best_model.pkl      # final tuned model
├── label_encoder.pkl
├── drift_current.html
├── drift_stress.html
└── shap_per_class.png
```

## Getting started

```bash
python -m venv mlops
mlops\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter lab
```

To browse experiment runs, start the MLflow UI from inside `project 1/`:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Author

Vishnu Konjeti

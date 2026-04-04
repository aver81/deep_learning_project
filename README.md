# A Comparative Study of Probabilistic Electricity Load Forecasting Using Baseline, LSTM, and Transformer Models

**DATA 612: Deep Learning -- University of Maryland, College Park**

**Group 2:** Aayush Verma, Kanishk Kaul, Rishi Koushik Sridharan, Samarth Singh

---

## Overview

This project applies a Transformer-based deep learning model to predict electricity consumption using the [UCI Electricity Load Diagrams 2011-2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) dataset. The Transformer outputs probabilistic forecasts (10th, 50th, and 90th percentile predictions) for a 24-hour horizon, trained with Quantile Loss (Pinball Loss). Performance is compared against naive baselines, seasonal baselines, and LSTM models.

## Repository Structure

```
deep_learning_project/
└── experimentation/
    ├── data_utils.ipynb       # Data pipeline (download, clean, features, split, normalize)
    ├── baselines.ipynb        # Naive baselines + LSTM models
    ├── transformers.ipynb     # Probabilistic Transformer model
    └── evaluation.ipynb       # Evaluation metrics, comparison, and visualizations
```

### File Descriptions

**`data_utils.ipynb`** -- Data Engineering & Preprocessing
- Downloads the UCI Electricity dataset (370 customers, 15-min intervals, 2011-2014)
- Aggregates customer-level readings into a single hourly total load series
- Creates cyclical time features (hour_sin/cos, dayofweek_sin/cos, month_sin/cos, is_weekend)
- Splits data chronologically (70% train / 15% val / 15% test)
- Normalizes using StandardScaler fitted on training data only
- **Output variables:** `train_scaled`, `val_scaled`, `test_scaled`, `scaler`, `df_features`

**`baselines.ipynb`** -- Baseline Models
- Clones the repo and runs `data_utils.ipynb` via `%run` to load the preprocessed data
- Creates sliding window sequences (168h input, 24h forecast)
- Implements three naive baselines:
  - Naive Persistence (repeat last 24h)
  - Seasonal Naive Day (same hour yesterday, lag=24)
  - Seasonal Naive Week (same hour last week, lag=168)
- Trains two LSTM models (MSE loss, point forecast):
  - Univariate LSTM (load only, input_size=1)
  - Multivariate LSTM (7 features, input_size=7)
- **Output variables:** `naive_persistence_preds_scaled`, `seasonal_naive_day_preds_scaled`, `seasonal_naive_week_preds_scaled`, `univariate_lstm_preds_scaled`, `multivariate_lstm_preds_scaled`, `y_test_scaled`

**`transformers.ipynb`** -- Probabilistic Transformer
- Clones the repo and runs `data_utils.ipynb` via `%run` to load the preprocessed data
- Implements the Transformer encoder architecture:
  - Input projection (7 features to d_model=64)
  - Sinusoidal positional encoding
  - 2-layer Transformer encoder (4 heads, FFN=128)
  - Output head producing 24x3 quantile predictions (q10, q50, q90)
- Trains with Quantile Loss (Pinball Loss) for 30 epochs with LR scheduling
- **Output variables:** `model` (trained Transformer), `train_losses`, `val_losses`

**`evaluation.ipynb`** -- Evaluation & Comparison
- Runs `baselines.ipynb` and `transformers.ipynb` via `%run` to get all trained models and predictions
- Generates Transformer test predictions (extracts q50 median as point forecast)
- Inverse-transforms all predictions to original scale
- Implements evaluation metrics from scratch: MAE, RMSE, MAPE
- Produces comparison table, improvement analysis, and visualizations:
  - Lollipop chart (models ranked by metric)
  - 24-hour forecast overlay (all models vs actual)
  - Horizon-wise error analysis (h=1 to h=24)
  - Transformer training/validation loss curve

## How to Run

### Prerequisites

- Google Colab (recommended, GPU runtime)
- Python 3.10+
- PyTorch, NumPy, Pandas, Matplotlib, scikit-learn

### Step-by-Step

1.  Step-by-Step

 **Clone the repository**
```bash
   git clone https://github.com/samarthsingh1/deep_learning_project.git
   cd deep_learning_project/experimentation
```

   Or on Google Colab:
```python
   !git clone https://github.com/samarthsingh1/deep_learning_project.git
```
2.  **Open any notebook in Google Colab**

   Each notebook is self-contained. It clones this repo and runs its dependencies automatically via `%run`.

3. **Run `evaluation.ipynb` for the full pipeline**

   This is the main notebook that chains everything together:
   ```
   evaluation.ipynb
     └── %run baselines.py
           └── %run data_utils.py    (downloads data, preprocesses)
           └── trains LSTMs, generates baseline predictions
     └── %run transformers.py
           └── %run data_utils.py    (downloads data, preprocesses)
           └── trains Transformer
     └── generates Transformer test predictions
     └── computes all metrics and visualizations
   ```

   Simply open `evaluation.ipynb` in Colab, set runtime to GPU, and **Run All**.

4. **Or run individual notebooks**

   - Run `data_utils.ipynb` alone to explore the dataset and preprocessing
   - Run `baselines.ipynb` to train and evaluate only the baselines and LSTMs
   - Run `transformers.ipynb` to train only the Transformer model

### Runtime Estimates (Colab GPU)

| Notebook | Approximate Runtime |
|----------|-------------------|
| `data_utils.ipynb` | ~3 min (dataset download) |
| `baselines.ipynb` | ~10 min (download + LSTM training) |
| `transformers.ipynb` | ~15 min (download + Transformer training) |
| `evaluation.ipynb` | ~30 min (runs both baselines + transformer + evaluation) |

### Notes

- The dataset (~250 MB) is downloaded automatically from the UCI repository on each run. A stable internet connection is required.
- The first cells in `baselines.ipynb` and `transformers.ipynb` clone this repo and convert `data_utils.ipynb` to a `.py` script using `jupyter nbconvert`, then execute it with `%run`. This is how the shared data pipeline is reused across notebooks.
- All models use the same chronological 70/15/15 train/val/test split to ensure fair comparison.
- Random seeds are set (`set_seed(42)`) for reproducibility, though minor variations may occur across different GPU hardware.

## Results Summary

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| Naive Persistence | 35,515 | 58,203 | 3.82 |
| Seasonal Naive Day | 35,515 | 58,203 | 3.82 |
| Seasonal Naive Week | 59,839 | 95,700 | 5.79 |
| Univariate LSTM | 37,442 | 52,527 | 4.23 |
| Multivariate LSTM | 34,356 | 48,108 | 3.81 |
| **Transformer (q50)** | **33,356** | **48,018** | **3.66** |

## References

1. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
2. UCI ML Repository. Electricity Load Diagrams 2011-2014.
3. Hochreiter, S. and Schmidhuber, J. "Long Short-Term Memory." Neural Computation, 1997.
4. Koenker, R. and Bassett, G. "Regression Quantiles." Econometrica, 1978.
5. Paszke, A., et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019.

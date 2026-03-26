# trunk-ml
### Business Rehearsal Outcome Predictor
*A machine learning layer for Trunk's Business Rehearsal feature.*
Joey Quatela · Augmented Intelligence · Spring 2026

## Overview
This project builds an ML prediction layer that forecasts downstream business
impacts — inventory, cash flow, and order velocity — as confidence-ranged
intervals when a founder creates a scenario branch in Trunk's Business Rehearsal.

## Research Goals
1. **Algorithm comparison** — XGBoost vs. Transformer vs. linear regression baseline
2. **Business category analysis** — does a global model generalize, or do retail, apparel, and B2B need separate models?

## Project Structure
| Folder | Purpose |
|---|---|
| `data/` | Raw inputs, processed datasets, ingestion scripts |
| `features/` | Feature engineering pipeline |
| `models/` | Baseline, XGBoost, and Transformer model code |
| `evaluation/` | Metrics, calibration plots, SHAP analysis |
| `notebooks/` | EDA, per-model experiments, comparison writeup |

## 6-Week Timeline
| Week | Focus |
|---|---|
| 1 | Data pipeline & feature engineering |
| 2 | Linear regression baseline |
| 3 | XGBoost model |
| 4 | Transformer model + algorithm comparison |
| 5 | Business category analysis |
| 6 | Tuning, evaluation, final write-up |

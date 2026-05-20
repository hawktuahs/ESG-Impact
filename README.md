# ESG Impact on Investment Returns

**Impact of ESG Factors on Investment Returns: Cross-Regional Evidence from Panel Econometrics and Hybrid Machine Learning**

This repository contains the runnable modelling code for the ESG investment-return analysis. The manuscript and paper-generation files are kept outside the public repository.

## Repository Structure

- `src/esg_return_modeling.py` - reproducible modelling pipeline.
- `src/generate_research_figures.py` - creates the research figures from generated model outputs.
- `requirements.txt` - Python package requirements.
- `data/` - expected local location for the raw ESG-financial CSV. This folder is intentionally ignored by Git.
- `results/` - generated outputs from the modelling pipeline. This folder is intentionally ignored by Git.

## Data

The pipeline expects the raw ESG-financial panel as:

```text
data/Asia_us_eu_1.csv
```

Place the dataset at that path before running the script. The data file is not included in the public code repository.

## Reproducing Results

Create and activate a Python environment, then install the required libraries:

```bash
pip install pandas numpy scipy scikit-learn statsmodels linearmodels matplotlib seaborn python-docx xgboost lightgbm shap
```

Run the analysis:

```bash
python src/esg_return_modeling.py
```

Generate figures:

```bash
python src/generate_research_figures.py
```

## Method Summary

The analysis uses one-year-lagged ESG and financial predictors. Returns are adjusted by region-year and sector-year. Verified Fama-French regional five-factor files are downloaded from the Kenneth R. French Data Library and aggregated from monthly percent factors to annual decimal factors.

Models include clustered pooled OLS, firm-year fixed effects, Ridge, ElasticNet, LinearSVR, Random Forest, Gradient Boosting, XGBoost, LightGBM, Logistic Regression, LinearSVC, and a regularised multilayer perceptron. Predictive models are evaluated with walk-forward splits for 2021-2024 using RMSE, MAE, R-squared, direction accuracy, F1, AUC, and top-minus-bottom quintile spreads.

## Current Finding

The results support a restrained interpretation: ESG, especially the environmental pillar, has cross-sectional explanatory value for region-adjusted investment returns. Hybrid ML models add modest ranking information when ESG, financial variables, and Fama-French factor context are combined. The evidence does not support an ESG-only annual alpha claim.

Generated model outputs are written to `results/tables/`; generated figures are written to `results/figures/`.

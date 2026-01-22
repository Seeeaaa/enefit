# Enefit - Predict Energy Behavior of Prosumers
**Predict Prosumer Energy Patterns and Minimize Imbalance Costs.**

This repository contains a pet project based on the Kaggle competition **[Enefit — Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers)**. The goal is to forecast both consumption and production of electrical energy for prosumers, hour by hour, minimizing imbalance costs. Prosumers are located in different counties (Estonia), have different contract types, and may be residential or commercial. All analysis, baselines comparisons and tuning steps are provided in Jupyter notebooks, viewable directly on GitHub.

The number of prosumers is rapidly increasing, creating challenges for grid stability and imbalance costs. Forecasting of consumption and production can reduce operational expenses and improve renewable integration. This project addresses these challenges by building and tuning ML models on the Kaggle "Enefit - Predict Energy Behavior of Prosumers" competition dataset.

## Notebooks

**`0. Competition Overview.ipynb`** - all information about competition from the host.  
**`1. Exploratory Data Analysis.ipynb`** - data analysis: missing values, time series consistency, data merging, feature engineering, and visualizations.  
**`2. Baseline Comparison.ipynb`** - comparison of baseline models.  
**`3. Hyperparameter Optimization.ipynb`** - hyperparameter search for the selected model using Optuna. Final evaluation metrics and feature importance analysis.  

All notebooks include commentary, tables, and plots to illustrate steps. Simply click on any `.ipynb` file in this repository to view outputs on GitHub.

## Directory Structure
```
├── models
├── notebooks
│ ├── 0. Competition Overview.ipynb
│ ├── 1. Exploratory Data Analysis.ipynb
│ ├── 2. Baseline Comparison.ipynb
│ └── 3. Hyperparameter Optimization.ipynb
├── optuna_db
├── utils
│ ├── __init__.py
│ ├── feature_engineering.py
│ ├── loading.py
│ ├── merging.py
│ └── preprocessing.py
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

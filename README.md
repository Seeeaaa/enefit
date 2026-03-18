# Enefit — Predict Energy Behavior of Prosumers
**Predict Prosumer Energy Patterns and Minimize Imbalance Costs.**

This repository contains a pet project based on the Kaggle competition **[Enefit — Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers)**. The goal is to forecast both consumption and production of electrical energy for prosumers, hour by hour, minimizing imbalance costs. Prosumers are located in different counties of Estonia, have different contract types, and may be residential or commercial.

The number of prosumers is rapidly increasing, creating challenges for grid stability and imbalance costs. Forecasting consumption and production can reduce operational expenses and improve renewable integration. This project addresses these challenges through exploratory data analysis (Python, SQL), baseline comparison, and ML model hyperparameter-tuning. All preparation steps, analysis, ML pipeline steps are provided via SQL files and Jupyter notebooks, viewable directly on GitHub.

## Contents
- [Notebooks](#notebooks)
- [SQL](#sql)
- [Directory Structure](#directory-structure)
- [Stack](#stack)

## Notebooks
**[`0. Competition Overview.ipynb`](notebooks/0.%20Competition%20Overview.ipynb)** - competition description, features, and evaluation metric.
<!-- **`0. Competition Overview.ipynb`** - all information about competition from the host.   -->
**[`1. Exploratory Data Analysis.ipynb`](notebooks/1.%20Exploratory%20Data%20Analysis.ipynb)** - missing values, data types, time series consistency, data merging, feature engineering, and visualizations.
<!-- **`1. Exploratory Data Analysis.ipynb`** - data analysis: missing values, time series consistency, data merging, feature engineering, and visualizations.   -->
**[`1.1. SQL EDA.ipynb`](notebooks/1.1.%20SQL%20EDA.ipynb)** - SQL-based exploratory analysis: data stored in a PostgreSQL service running via Docker Compose, multi-table JOINs, CTEs, window functions (moving averages, MoM change), aggregations by segment, installed capacity CAGR (53% annually, reflecting rapid growth of prosumer solar panels installations in Estonia).

**[`2. Baseline Comparison.ipynb`](notebooks/2.%20Baseline%20Comparison.ipynb)** - comparison of XGBoost, LightGBM, and CatBoost baseline models.
<!-- **`2. Baseline Comparison.ipynb`** - comparison of baseline models.   -->
**[`3. Hyperparameter Optimization.ipynb`](notebooks/3.%20Hyperparameter%20Optimization.ipynb)** - Optuna-based hyperparameter search for XGBoost, GPU-accelerated training and final evaluation metrics. Test MAE reduced from 72.8 to 71.5.
<!-- **`3. Hyperparameter Optimization.ipynb`** - hyperparameter search for the selected model using Optuna. Final evaluation metrics and feature importance analysis.   -->

All notebooks include commentary, tables, and plots to illustrate steps. Simply click on any `.ipynb` file in this repository to view outputs on GitHub.

## SQL
PostgreSQL container connection defined in `docker-compose.yml`. Interaction from the JLPE container uses `psql` (for running DDL and ETL scripts) and SQLAlchemy + psycopg (for analytical queries in the notebook).

**[`sql/ddl.sql`](sql/ddl.sql)** - table definitions for `train`, `client`, `gas_prices` and `electricity_prices`.

**[`sql/load.sql`](sql/load.sql)** - truncates and reloads all source tables from CSV files via `\copy`. Idempotent.

**[`sql/transform.sql`](sql/transform.sql)** - creates `merged` table by joining all source tables with CTEs for preprocessing (e.g. electricity price date shift).

**[`sql/indexes.sql`](sql/indexes.sql)** - index on `train(datetime)` for time-based filtering.

## Directory Structure
```
├── dashboard
├── notebooks
│ ├── 0. Competition Overview.ipynb
│ ├── 1. Exploratory Data Analysis.ipynb
│ ├── 1.1. SQL EDA.ipynb
│ ├── 2. Baseline Comparison.ipynb
│ └── 3. Hyperparameter Optimization.ipynb
├── optuna_db
├── sql
│ ├── ddl.sql
│ ├── indexes.sql
│ ├── load.sql
│ └── transform.sql
├── utils
│ ├── __init__.py
│ ├── data_pipeline.py
│ ├── feature_engineering.py
│ ├── loading.py
│ ├── merging.py
│ └── preprocessing.py
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Stack
Python, SQL, PostgreSQL, Docker Compose, Pandas, NumPy, SciPy, Matplotlib, Seaborn, Plotly, Power BI, Power Query, DAX, Scikit-learn, XGBoost, LightGBM, CatBoost, Optuna.
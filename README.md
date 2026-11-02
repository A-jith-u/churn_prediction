# Telecom Customer Churn Prediction

A small data science project for cleaning Telco customer data, exploring churn patterns, and preparing machine-learning features.

## Project Structure

- `clean_data.py` cleans the raw dataset and writes `data/processed/telco_clean.csv`.
- `notebooks/01_cleaning_and_eda.ipynb` performs cleaning checks and exploratory analysis.
- `notebooks/02_modeling.ipynb` engineers features, splits the data, and builds the preprocessing pipeline.
- `data/raw/` contains the source dataset.
- `data/processed/` contains the cleaned dataset.
- `reports/figures/` stores generated figures.

## Setup

Create or activate the project virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the notebook and data-science dependencies:

```powershell
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
```

In VS Code, select this interpreter as the notebook kernel:

```text
venv\Scripts\python.exe
```

## Run the Workflow

From the project root, clean the raw data:

```powershell
python clean_data.py
```

Then open the notebooks in order:

1. `notebooks/01_cleaning_and_eda.ipynb`
2. `notebooks/02_modeling.ipynb`

The notebooks expect the processed CSV at `data/processed/telco_clean.csv`.

## Notes

The cleaning step removes target leakage, redundant fields, and non-predictive identifiers before creating the binary `churn` target.

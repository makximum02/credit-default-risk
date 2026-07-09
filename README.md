# Credit Default Risk - Analysis & Modeling

Predicting which borrowers are likely to default, using the "Give Me Some Credit" Kaggle dataset (~150,000 borrowers).
Built end-to-end: SQL data cleaning, exploration analysis, and an interpretable logistic regression model.


## Key Results

- **ROC-AUC: 0.84** on a held-out test of 30,000 borrowers
- **Credit utilization** is the strongest default predictor(odds ratio 2.18), followed by past-due history(1.81)
- Found and fixed **multicolineaity**: three correlated past-due features (r = 0.74-0.86) produced an implausible negative coefficient; engineered them into one stable feature
- **Threshold tuning:** the default 0.50 cutoff caught only 6% of defaulters despite 93% accuracy (class imbalance trap). Lowering it to 0.10 raised recall to 66% _the optimal cutoff is a business decision balancing missed defaults vs false alarms.


## Approach

1. **Load** - Python script reads the raw CSV into SQLITE
2. **Clean (SQL)** - capped sentinel code (96/98) and garbage ratios, filtered impossible ages, imputed missing income(median) with a "was missing"flag
3. **Explore (SQL)** - default rates by utilization, age, past-due history, debt ratio; two-factor interaction analysis
4. **Model (Python)** - standardized features, stratified 80/20 split, logistic regression, imbalance-aware evaluation


## Tech Stack

SQL (SQLite) . Python (pandas, scikit-learn) . Git


## Repository

- `src/load_to+sqlite.py` - data loading pipeline
- `model.py` - full modeling pipeline (split, scale,train,evaluate)
- `FINDINGS.md` - complete write-up with all results and decisions


## How to Run

1. Download `cs-training.csv` from the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) kaggle competition into `data/raw/`
2. `python3 src/load_to_sqlite.py` -builds the database
3. Run the cleaning SQL, then `python3 model.py`
*Data not included in this repository (Kaggle terms + repo hygiene).*

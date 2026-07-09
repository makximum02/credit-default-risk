import pandas as pd
import sqlite3
import os
df = pd.read_csv("data/raw/cs-training.csv")
print("Rows loaded:", len(df))
df = df.drop(columns=["Unnamed: 0"])
df = df.rename(columns={
    "SeriousDlqin2yrs": "serious_dlq_2yrs",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "NumberOfTime30-59DaysPastDueNotWorse": "num_30_59_past_due",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "num_open_credit_lines",
    "NumberOfTimes90DaysLate": "num_90_days_late",
    "NumberRealEstateLoansOrLines": "num_real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "num_60_89_past_due",
    "NumberOfDependents": "num_dependents"
})
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/credit.db")
df.to_sql("raw_credit", conn, if_exists="replace", index=False)
conn.close()

print("Done. Loaded", len(df), "rows into data/credit.db")

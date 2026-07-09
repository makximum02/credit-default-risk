import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

conn = sqlite3.connect("data/credit.db")
df = pd.read_sql("SELECT * FROM credit_clean", conn)
conn.close()

print("Loaded", len(df), "rows")
print(df.columns.tolist())

# Separate the target (what we predict) from the features (what we predict FROM)
y = df["serious_dlq_2yrs"]
X = df.drop(columns=["serious_dlq_2yrs"])

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set:", len(X_train), "borrowers")
print("Test set:", len(X_test), "borrowers")
print("Default rate - train:", round(y_train.mean() * 100, 2), "%")
print("Default rate - test:", round(y_test.mean() * 100, 2), "%")

# Scale features: fit the scaler on TRAINING data only, then apply to both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled.")

# Train logistic regression on the scaled training data
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print("Model trained.")
print("Intercept (b0):", round(model.intercept_[0], 4))

# Pull out the coefficients and pair them with feature names
import pandas as pd
import numpy as np
coefs = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
})
coefs["odds_ratio"] = np.exp(coefs["coefficient"])
coefs = coefs.sort_values("coefficient", ascending=False)

print("\n--- Coefficients (sorted, strongest positive first) ---")
print(coefs.to_string(index=False))

# Make predictions on the TEST set (data the model never saw)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ROC-AUC: how well the model separates defaulters from non-defaulters
auc = roc_auc_score(y_test, y_prob)
print("\n=== MODEL EVALUATION (on unseen test data) ===")
print("ROC-AUC:", round(auc, 4))

# Confusion matrix: the four boxes
print("\nConfusion matrix:")
print("               Pred: No   Pred: Yes")
cm = confusion_matrix(y_test, y_pred)
print(f"Actual No:    {cm[0][0]:>8}   {cm[0][1]:>8}")
print(f"Actual Yes:   {cm[1][0]:>8}   {cm[1][1]:>8}")

# Precision, recall, F1 per class
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["No default", "Default"]))

# Test several thresholds to see the precision/recall tradeoff
from sklearn.metrics import precision_score, recall_score

print("\n=== THRESHOLD TUNING ===")
print(f"{'Threshold':>10} {'Caught':>8} {'Missed':>8} {'FalseAlarm':>11} {'Precision':>10} {'Recall':>8}")

for t in [0.50, 0.30, 0.20, 0.15, 0.10, 0.07]:
    preds = (y_prob >= t).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()   # caught defaulters
    fn = ((preds == 0) & (y_test == 1)).sum()   # missed defaulters
    fp = ((preds == 1) & (y_test == 0)).sum()   # false alarms
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds)
    print(f"{t:>10.2f} {tp:>8} {fn:>8} {fp:>11} {prec:>10.3f} {rec:>8.3f}")



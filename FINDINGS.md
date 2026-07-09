# Credit Default Ridk-Exploration Findings

**Dataset:**Give Me some Credit(Kaggle), ~1500,000 borrowers
**Target:** serious_dlq_2yrs (1 is = to serious elinquency within 2 years)
**The overall default rate (baseline):** 6.68%

## Summary
We explored which borrower characteristics predicts default. Utilization, past-due history, and age are strong, clean predictors.Debt ratio is usable but it's partially contaminated by data-quality isseus. Utilization and age interact: risk compounds when both are unfavorable.

## Key Findings (default rate by factor)

### Credit Utilization is the strongest, cleanest signal)
Default rate climbs steadily with utilization:
-Under 10% used:1.81%
-70-100% used:17.71%
-Over 100% used:37.18%
Crosses the 6.68% baseline around 50% utilization. Measures current financial strain. ~20x risk difference across the range.

### Past-Due History (90+ days late)
-Never late: 4.63% (below baseline = safer than average)
-Ever late (1+times): jumps to 34-67%
-8338 borrowers had been 90+ late: they default at 41.64% which is  6times more the baseline measures.

### Age
Default rate falls steadily as age rises ( a downward staircase):
-Under 30: 11.73%
-70+:2.32%
Crosses baseline around age 50. older borrowers are safer.

### Debt Ratio(partially usable)
Real signal in the mid-range( it rises from 5.6% to 10.7% across 0.2-1.0),
but the "over 1.0" band is countaminated by missing and 0 income producing invalid pre-calculated ratios.

### Interaction: Utilization x Age
Combining the two strongest predictors: 
-Safet profile (60+, low utilizatio):1.38%
-Riskiest profile(under 40, high utilization): 22.16%
~16x spread. The two factors compound rather than simply add: high utilization hits younger borrowers harder than older ones.

##Data-Quality Issues Handled
-Past_due sentinel codes(96,98): not real counts; capped at 20. 
-Utilization/ debt ratio: garbage values in the thousands; capped at 2.
-Impossible ages (under 10):1 row, filtered out.
-Missing income: 29,731 rows (~20%); imputed with median(5,400) and flagged with income_was missing colum.
-Zero income: 1,634 rows(~1%); left as-is, noted as a limitation.

## Next Steps
-We will Build a logistic regression model on the cleaned data.
-Evaluate with ROC-AUC, precision, and consider the target is imbalanced(means the two outcomes we are predicting are very unequal.
-Add a gradient-boosted model to compare

## MODELING (WEEK 2)
Built a logistic regression model to predict default probability from the cleaned features.

### Setup
- 80/20 train/test split, stratified to preserve the 6.68% default rate.
- Features standardized (mean 0, std 1) so coefficients are comparable.
- Trained on 119,999 borrowers, tested on 30,000 unseen borriwers.
 
### Feature Engineering: Multicollinearity Fix
The three past_due columns (30-59, 60-89, 90+ days) were highly correlated(0.74 to 0.86), which produced unstable, implausibly NEGATIVE coefficient on the 60-90 feature. Combined them into a single "total_past_due" feature. This stabilized the coefficients ( the combined feature came out sensibly positive) and made it 2ndstrongest predictor.

### Coefficients (which factors drive default risk)
Signs matched the week 1 exploration:
- Utilization: strongest driver (odds ratio 2.18)
- Total past-due history: 2nd strongest (odds ratio 1.81)
- Age: protective (older=lower risk)
- Income: protective (higher = lower risk)

### Performance
- ROC-AUC: 0.84 (strong; the model ranks risk well)
- At the default 0.50 threshold: 93% accuracy but only 6% recall.
- Accuracy was misleading due to class imbalance.

### Threshold Tuning (the key insight)
Because defaults are rare, the 0.50 threshold missed 94% of defaulters.
Lowering the threshold trades precision for recall:
- Threshold 0.50: caught 6% of defaulters
- Threshold 0.10: caught 66% of defaulters (more false positives)
The optimal threshold is a business decision, set by the relative cost of a missed default vs. a false alarm.

###Skills Demonstrated
SQL (cleaning, aggregate, feature engineering), Python (pandas, scikit-learn), logistic regression, handling class imbalance, multicollinearity diagnosis, modelevaluation(ROC-AUC, precision/recall), threshold tuning.

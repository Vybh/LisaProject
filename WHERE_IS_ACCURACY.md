# Where to Find ACCURACY (Acc) Metric

## 🎯 Quick Answer

**ACCURACY is shown in:** `results/test_results.csv`

---

## 📊 Test Set Accuracy (Final Evaluation)

**File:** `/results/test_results.csv`

```
Model,Accuracy,Precision,Recall,F1_Score
↓
SVM,0.8518518518518519,0.8902557469630641,0.8632625272331155,0.864566221735908
                      ↑
                ACCURACY = 85.19%
```

### Final Test Accuracy Results:
| Model | **Accuracy (Acc)** |
|-------|:---:|
| **SVM** ⭐ | **85.19%** |
| kNN | 80.00% |
| XGBoost | 80.00% |
| RandomForest | 78.52% |

**What this means:** 
- SVM correctly classified **115 out of 135** test images
- Error rate: 14.81% (20 misclassified images)

---

## 📈 Cross-Validation Accuracy (Model Robustness)

**File:** `/results/cv_results.csv`

```
Model,Accuracy_Mean,Accuracy_Std,Precision_Mean,Recall_Mean,F1_Mean,Train_Acc_Mean
↓
SVM,0.8031746031746032,0.03736064220933277,0.8231942740495372,0.807433364847158,0.80403451888881771,0.9563492063492063
    ↑
    ACCURACY_MEAN = 80.32%
    ACCURACY_STD = ±3.74%
```

### Cross-Validation Accuracy (5-Fold):
| Model | **Accuracy_Mean** | **Accuracy_Std** | **Training Accuracy** |
|-------|:---:|:---:|:---:|
| **SVM** ⭐ | **80.32%** | ±3.74% | 95.63% |
| RandomForest | 77.62% | ±4.50% | 98.37% |
| kNN | 77.14% | ±4.30% | 98.37% |
| XGBoost | 77.14% | ±3.60% | 98.37% |

**What this means:**
- SVM's accuracy averaged 80.32% across 5 validation folds
- Standard deviation of ±3.74% shows consistency
- Training accuracy is 95.63%, so only ~15% overfitting gap

---

## 🔢 How Accuracy is Calculated

```python
Accuracy = Number of Correct Predictions / Total Predictions

SVM Example:
- Total test images: 135
- Correctly classified: 115
- Misclassified: 20
- Accuracy = 115 / 135 = 0.8519 = 85.19%
```

---

## 📂 All Locations Where Accuracy Appears

1. **`results/test_results.csv`** (Column 2)
   - Final test set accuracy for each model
   - This is the PRIMARY metric reported

2. **`results/cv_results.csv`** (Columns 2 & 3)
   - `Accuracy_Mean`: Average validation accuracy across 5 folds
   - `Accuracy_Std`: Standard deviation of accuracy
   - Also shows `Train_Acc_Mean`: Training accuracy to detect overfitting

3. **RESULTS_SUMMARY.md**
   - Human-readable summary with tables
   - Explains what accuracy means and how to interpret it

4. **Training Console Output**
   - Printed during cross-validation: `Acc: 0.8032 (±0.0374)`
   - Shows real-time validation accuracy per model

---

## 🎓 Understanding Accuracy in Context

### Test Accuracy vs Training Accuracy

| Model | Train Acc | Test Acc | Gap | Interpretation |
|-------|:---:|:---:|:---:|:---|
| SVM | 95.63% | 85.19% | 10.44% | Good! Healthy gap |
| RandomForest | 98.37% | 78.52% | 19.85% | Some overfitting |
| kNN | 98.37% | 80.00% | 18.37% | Some overfitting |
| XGBoost | 98.37% | 80.00% | 18.37% | Some overfitting |

**SVM has the best generalization** (smallest gap between train/test)

---

## 🎯 Comparison with Other Metrics

| Metric | Best For | Formula |
|--------|----------|---------|
| **Accuracy** | Overall correctness | (TP+TN)/(TP+TN+FP+FN) |
| Precision | Minimizing false positives | TP/(TP+FP) |
| Recall | Minimizing false negatives | TP/(TP+FN) |
| F1-Score | Balanced classification | 2·(Precision·Recall)/(Precision+Recall) |

**For this project:** Accuracy is the primary metric (balanced classes, equal cost of errors)

---

## 📊 Visual Representation

```
SVM Test Performance:
Total: 135 images
├── Correct: 115 images (85.19%) ✓
└── Wrong: 20 images (14.81%) ✗

RandomForest Test Performance:
Total: 135 images
├── Correct: 106 images (78.52%) ✓
└── Wrong: 29 images (21.48%) ✗

kNN Test Performance:
Total: 135 images
├── Correct: 108 images (80.00%) ✓
└── Wrong: 27 images (20.00%) ✗

XGBoost Test Performance:
Total: 135 images
├── Correct: 108 images (80.00%) ✓
└── Wrong: 27 images (20.00%) ✗
```

---

## 💾 How to Read the CSV Files

### test_results.csv
```python
import pandas as pd

df = pd.read_csv('results/test_results.csv')
print(df)

# Output:
#           Model  Accuracy  Precision    Recall  F1_Score
# 0           SVM       0.85       0.89      0.86      0.86
# 1  RandomForest       0.79       0.82      0.75      0.77
# 2           kNN       0.80       0.80      0.78      0.78
# 3       XGBoost       0.80       0.80      0.76      0.78

# Access specific accuracy:
svm_accuracy = df.loc[df['Model'] == 'SVM', 'Accuracy'].values[0]
print(f"SVM Accuracy: {svm_accuracy:.2%}")  # Output: SVM Accuracy: 85.19%
```

### cv_results.csv
```python
import pandas as pd

df = pd.read_csv('results/cv_results.csv')
print(df[['Model', 'Accuracy_Mean', 'Accuracy_Std', 'Train_Acc_Mean']])

# Output:
#        Model  Accuracy_Mean  Accuracy_Std  Train_Acc_Mean
# 0       SVM            0.80           0.04            0.96
# 1  RandomForest         0.78           0.05            0.98
# 2        kNN            0.77           0.04            0.98
# 3    XGBoost            0.77           0.04            0.98
```

---

## ✅ Summary

- ✅ **Accuracy IS shown** in `results/test_results.csv` (2nd column)
- ✅ **Best accuracy: 85.19%** (SVM model)
- ✅ **Cross-validation accuracy: 80.32%** (5-fold average)
- ✅ **Clear comparison** across all 4 models
- ✅ **Detailed interpretation** provided in multiple files

The accuracy metric is prominently displayed and can be found in the CSV files and RESULTS_SUMMARY.md!


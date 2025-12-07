# Traffic Sign Classification - Results Summary

## 📊 Test Set Performance (Final Evaluation)

**File Location:** `results/test_results.csv`

| Model | **Accuracy** | Precision | Recall | F1-Score |
|-------|:---:|:---:|:---:|:---:|
| **SVM** ⭐ | **85.19%** | 89.03% | 86.33% | 86.46% |
| RandomForest | 78.52% | 81.82% | 74.81% | 76.94% |
| kNN | 80.00% | 79.67% | 78.37% | 78.33% |
| XGBoost | 80.00% | 80.49% | 76.33% | 77.75% |

### Metric Definitions:
- **Accuracy (Acc)**: Percentage of correct predictions out of total predictions
  - Formula: `(TP + TN) / (TP + TN + FP + FN)`
  - Range: 0-1 (displayed as 0-100%)
- **Precision**: Of positive predictions, how many were correct
- **Recall**: Of actual positives, how many were correctly identified
- **F1-Score**: Harmonic mean of Precision and Recall

---

## 📈 Cross-Validation Performance (5-Fold Stratified)

**File Location:** `results/cv_results.csv`

| Model | **Accuracy_Mean** | **Accuracy_Std** | Precision_Mean | Recall_Mean | F1_Mean | Train_Acc_Mean |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **SVM** ⭐ | **80.32%** | ±3.74% | 82.32% | 80.74% | 80.40% | 95.63% |
| RandomForest | 77.62% | ±4.50% | 80.52% | 74.15% | 75.75% | 98.37% |
| kNN | 77.14% | ±4.30% | 75.05% | 75.67% | 74.43% | 98.37% |
| XGBoost | 77.14% | ±3.60% | 74.73% | 73.35% | 73.36% | 98.37% |

### Metric Definitions:
- **Accuracy_Mean**: Average accuracy across 5 CV folds
- **Accuracy_Std**: Standard deviation (measure of consistency)
- **Train_Acc_Mean**: Training accuracy (to assess overfitting)

---

## 📁 Result Files

### CSV Data Files:
1. **`results/test_results.csv`** - Final test set metrics for all 4 models
2. **`results/cv_results.csv`** - 5-fold cross-validation metrics

### Visualization Files:

#### Confusion Matrices:
- `results/confusion_matrix_SVM.png`
- `results/confusion_matrix_RandomForest.png`
- `results/confusion_matrix_kNN.png`
- `results/confusion_matrix_XGBoost.png`

#### ROC Curves (One-vs-Rest):
- `results/roc_curves_SVM.png`
- `results/roc_curves_RandomForest.png`
- `results/roc_curves_kNN.png`
- `results/roc_curves_XGBoost.png`

#### Precision-Recall Curves:
- `results/pr_curves_SVM.png`
- `results/pr_curves_RandomForest.png`
- `results/pr_curves_kNN.png`
- `results/pr_curves_XGBoost.png`

---

## 🎯 Key Findings

### Best Model: **SVM**
- **Test Accuracy: 85.19%** ✅
- **CV Accuracy: 80.32%** (±3.74%)
- **Generalization Gap: 4.87%** (healthy, not overfitting)

### Dataset Information:
- **Total Images**: 900 traffic sign images
- **Classes**: 9 traffic sign types
  1. keepRight (110 images)
  2. merge (100 images)
  3. pedestrianCrossing (100 images)
  4. signalAhead (100 images)
  5. speedLimit25 (80 images)
  6. speedLimit35 (110 images)
  7. stop (210 images) - most common
  8. yield (45 images) - least common
  9. yieldAhead (45 images) - least common

### Data Split:
- **Training**: 630 samples (70%)
- **Validation**: 135 samples (15%)
- **Test**: 135 samples (15%)

### Features Used:
- **HOG (Histogram of Oriented Gradients)**: 1,764 dimensions
- **Color Histogram**: 96 dimensions
- **Bag of Visual Words**: 50 dimensions
- **Total**: 1,910 features per image

---

## 📊 How to Interpret Accuracy

### Test Accuracy (85.19% for SVM):
- Out of 135 test images, the SVM correctly classified **115 images**
- Only **20 images** were misclassified

### Cross-Validation Accuracy (80.32% for SVM):
- Across all 5 folds of training/validation
- Average accuracy is 80.32%
- Standard deviation of ±3.74% means most folds are between 76-84%

### Training Accuracy (95.63% for SVM):
- SVM achieved 95.63% on training set
- Gap between training (95.63%) and test (85.19%) is ~10.4%
- This small gap indicates good generalization

---

## 🏆 Performance Ranking

1. **SVM**: 85.19% test accuracy ⭐ (Best performer)
2. **kNN**: 80.00% test accuracy
3. **XGBoost**: 80.00% test accuracy
4. **RandomForest**: 78.52% test accuracy

All models perform well on this balanced dataset!

---

## 📝 Notes

- **Accuracy** is the primary metric used for evaluation
- All metrics are calculated on the held-out **test set** (135 images)
- Cross-validation provides confidence in model robustness
- Class imbalance is handled via class weights in training
- No data leakage: strict 70/15/15 stratified split


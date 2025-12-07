# Project Setup and Execution Guide

## Quick Start (5 minutes)

### 1. Environment Setup

```bash
cd /Users/vybhavreddy/Desktop/tinylisaproject

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Ensure your LISATS dataset is organized as:
```
/Users/vybhavreddy/Desktop/LISATS/
├── Stop/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Yield/
│   ├── image1.jpg
│   └── ...
└── [other_classes]/
```

### 3. Run the Complete Pipeline

**Option A: Use main training script**
```bash
cd src
python train.py --data_path /Users/vybhavreddy/Desktop/LISATS
```

**Option B: Use Jupyter notebook** (Interactive)
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## Project Structure Overview

```
tinylisaproject/
│
├── src/                          # Core source code
│   ├── __init__.py
│   ├── datasets.py              # TrafficSignDataset class
│   │   ├── load_data()          # Load images from directories
│   │   ├── stratified_split()   # 70/15/15 split
│   │   └── get_class_weights()  # Handle imbalance
│   │
│   ├── features.py              # FeatureExtractor class
│   │   ├── extract_hog()        # HOG features (1764 dims)
│   │   ├── extract_color_histogram()  # ColorHist (96 dims)
│   │   ├── fit_bovw()           # Fit BoVW clustering
│   │   ├── extract_bovw()       # BoVW features (100 dims)
│   │   ├── extract_all_features()     # Concatenate all
│   │   ├── fit_scaler()         # Fit StandardScaler
│   │   └── scale_features()     # Normalize
│   │
│   ├── models.py                # ModelTrainer class
│   │   ├── get_models()         # Initialize 4 algorithms
│   │   ├── cross_validate_models()    # 5-fold CV
│   │   ├── train_final_models() # Train on full train set
│   │   ├── evaluate_models()    # Test set evaluation
│   │   └── get_feature_importance()   # Feature analysis
│   │
│   ├── evaluate.py              # Evaluator class
│   │   ├── plot_confusion_matrix()
│   │   ├── plot_roc_curves()
│   │   ├── plot_precision_recall()
│   │   ├── plot_calibration_curve()
│   │   ├── plot_model_comparison()
│   │   └── plot_learning_curves()
│   │
│   └── train.py                 # Main pipeline orchestrator
│       ├── Load config
│       ├── Load & split data
│       ├── Extract features
│       ├── 5-fold CV
│       ├── Train final models
│       ├── Evaluate
│       └── Save results
│
├── configs/
│   └── traffic_signs.yaml       # Configurable hyperparameters
│       ├── Dataset paths
│       ├── Feature options (HOG, ColorHist, BoVW)
│       ├── Model hyperparameters
│       └── Evaluation settings
│
├── notebooks/
│   └── EDA.ipynb               # Interactive Jupyter notebook
│       ├── Section 1: Data exploration
│       ├── Section 2: Stratified splitting
│       ├── Section 3: Feature extraction
│       ├── Section 4: Cross-validation
│       ├── Section 5: Test evaluation
│       ├── Section 6: Learning curves & feature importance
│       ├── Section 7: Robustness analysis
│       └── Section 8: Error analysis
│
├── models/                       # Trained model artifacts
│   ├── SVM.pkl
│   ├── RandomForest.pkl
│   ├── kNN.pkl
│   └── XGBoost.pkl
│
├── results/                      # Evaluation artifacts
│   ├── cv_results.csv           # Cross-validation scores
│   ├── test_results.csv         # Test metrics
│   ├── confusion_matrix_*.png    # 4 confusion matrices
│   ├── roc_curves_*.png          # ROC curves (one-vs-rest)
│   ├── pr_curves_*.png           # Precision-Recall curves
│   ├── calibration_*.png         # Calibration plots
│   ├── learning_curve_*.png      # Learning curves
│   ├── model_comparison_*.png    # Metric comparison charts
│   ├── summary.json              # Final results summary
│   └── [other visualizations]
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── REPORT_TEMPLATE.md           # Report structure (6-8 pages)
├── PRESENTATION_OUTLINE.md      # Slides outline (10-12 slides)
├── setup.sh                     # Automated setup script
└── SETUP_GUIDE.md              # This file
```

---

## Detailed Module Usage

### 1. datasets.py - Data Loading & Splitting

```python
from src.datasets import TrafficSignDataset

# Initialize
dataset = TrafficSignDataset(
    data_path='/Users/vybhavreddy/Desktop/LISATS',
    random_state=42
)

# Load all images
X, y = dataset.load_data()

# Stratified 70/15/15 split
X_train, X_val, X_test, y_train, y_val, y_test = dataset.stratified_split()

# Get class weights for imbalance handling
class_weights = dataset.get_class_weights(y_train)
```

### 2. features.py - Feature Extraction

```python
from src.features import FeatureExtractor

# Initialize
fe = FeatureExtractor(n_clusters=100, scaler_type='standard')

# Fit BoVW on training set
fe.fit_bovw(X_train)

# Extract all features
X_train_features = fe.extract_all_features(X_train)
X_test_features = fe.extract_all_features(X_test)

# Scale features
X_train_scaled = fe.fit_transform(X_train_features)
X_test_scaled = fe.scale_features(X_test_features)
```

**Feature Dimensions**:
- HOG: 1764 features
- ColorHist: 96 features (32 bins × 3 channels)
- BoVW: 100 features (100 clusters)
- **Total**: 1960 features

### 3. models.py - Model Training

```python
from src.models import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(random_state=42, n_splits=5)

# Cross-validation
cv_results = trainer.cross_validate_models(
    X_train_scaled, y_train, 
    class_weights=class_weights
)

# Train final models
trainer.train_final_models(
    X_train_scaled, y_train,
    class_weights=class_weights
)

# Evaluate on test set
test_results = trainer.evaluate_models(X_test_scaled, y_test)
```

**Models Compared**:
| Model | Hyperparameters |
|-------|-----------------|
| SVM | kernel=rbf, C=1.0, gamma=scale |
| Random Forest | n_estimators=100, max_depth=15 |
| k-NN | n_neighbors=5, weights=distance |
| XGBoost | n_estimators=100, max_depth=6, lr=0.1 |

### 4. evaluate.py - Visualization & Analysis

```python
from src.evaluate import Evaluator

evaluator = Evaluator(output_dir='results')

# Generate visualizations
evaluator.plot_confusion_matrix(y_test, y_pred, class_names, 'SVM')
evaluator.plot_roc_curves(y_test, y_proba, class_names, 'SVM')
evaluator.plot_precision_recall(y_test, y_proba, class_names, 'SVM')
evaluator.plot_calibration_curve(y_test, y_proba_max, 'SVM')
evaluator.plot_model_comparison(test_results, 'Accuracy')
```

---

## Configuration File (configs/traffic_signs.yaml)

Customize the pipeline by editing:

```yaml
# Dataset
dataset:
  path: "/Users/vybhavreddy/Desktop/LISATS"
  train_size: 0.7

# Features
features:
  bovw:
    n_clusters: 100

# Training
training:
  n_splits: 5
  models:
    random_forest:
      n_estimators: 100
      max_depth: 15
```

---

## Typical Execution Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Data Loading | 1-2 min | Depends on dataset size |
| BoVW Fitting | 2-3 min | SIFT extraction & clustering |
| Feature Extraction | 3-5 min | All 3 feature types on all data |
| 5-Fold CV | 2-3 min | Training 4 models × 5 folds |
| Final Training | 1 min | Train on full train set |
| Evaluation | <1 min | Test set predictions |
| Visualization | 1-2 min | Plot generation |
| **Total** | **~12-17 min** | For ~1000 images |

---

## Expected Output Files

### In `results/` directory:
- `cv_results.csv` - CV metrics for 4 models
- `test_results.csv` - Test metrics (Acc, Prec, Recall, F1)
- `confusion_matrix_SVM.png` - 4 confusion matrices
- `roc_curves_SVM.png` - ROC curves (multi-class one-vs-rest)
- `pr_curves_SVM.png` - Precision-Recall curves
- `calibration_SVM.png` - Calibration plots
- `model_comparison_Accuracy.png` - Bar chart comparison
- `summary.json` - Final results dictionary

### In `models/` directory:
- `SVM.pkl` - Trained SVM model
- `RandomForest.pkl` - Trained RF model
- `kNN.pkl` - Trained k-NN model
- `XGBoost.pkl` - Trained XGBoost model

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python opencv-contrib-python
```

### Issue: SIFT features not working
```bash
pip install --upgrade opencv-contrib-python
```

### Issue: Out of memory during feature extraction
- Reduce batch size in `extract_all_features()`
- Use subset of data for initial testing
- Consider downsampling images

### Issue: Dataset path not recognized
- Verify path: `ls /Users/vybhavreddy/Desktop/LISATS`
- Check subdirectories exist: `ls /Users/vybhavreddy/Desktop/LISATS/Stop`
- Update `--data_path` argument if using different location

---

## Performance Tips

1. **Faster Iteration**: Use subset of data first
   ```python
   X = X[:500]  # Test with 500 images
   y = y[:500]
   ```

2. **Parallel Processing**: Already enabled in:
   - Random Forest: `n_jobs=-1`
   - k-NN: `n_jobs=-1`
   - Model evaluation: `cross_validate` with `n_jobs=1` (to avoid nested parallelism)

3. **GPU Acceleration**: XGBoost supports GPU if available
   - Uncomment in `models.py`: `tree_method='gpu_hist'`

4. **Caching Features**: Save extracted features to avoid re-extraction
   ```python
   np.save('X_train_features.npy', X_train_features)
   ```

---

## Next Steps

1. **Run the pipeline**: Execute `python train.py`
2. **Analyze results**: Open `notebooks/EDA.ipynb` in Jupyter
3. **Generate report**: Write 6-8 page report using `REPORT_TEMPLATE.md`
4. **Create slides**: Prepare 10-12 slide presentation using `PRESENTATION_OUTLINE.md`
5. **Deploy model**: Package best model for production use

---

## References

- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## Project Timeline

- ✅ **Week 1**: Exploratory Data Analysis
- ✅ **Week 2**: Feature Engineering & Implementation
- ✅ **Week 3**: Model Training & CV
- 📋 **Week 4**: Evaluation & Robustness Analysis
- 📋 **Week 5**: Report Writing & Presentation

---

**Last Updated**: December 2024  
**Maintainer**: [Your Name]

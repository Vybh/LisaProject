# Traffic Sign Classification Project

A comprehensive machine learning pipeline for classifying traffic signs from dashcam footage using traditional ML algorithms and multiple feature types.

## Project Structure

```
tinylisaproject/
├── src/                          # Core source code
│   ├── __init__.py
│   ├── datasets.py              # Data loading & stratified splitting
│   ├── features.py              # HOG, ColorHist, BoVW extraction
│   ├── models.py                # SVM, RF, k-NN, XGBoost training
│   ├── evaluate.py              # Evaluation & visualization
│   └── train.py                 # Main training pipeline
├── configs/
│   └── traffic_signs.yaml       # Configuration file
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis
├── models/                      # Trained model pickles
├── results/                     # Results, visualizations, metrics
├── README.md                    # This file
└── REPORT_TEMPLATE.md          # Report structure
```

## Installation

### Requirements
- Python 3.8+
- pip

### Setup

```bash
# Clone or navigate to project directory
cd tinylisaproject

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Organize your LISATS dataset in the following structure:

```
/path/to/LISATS/
├── Stop/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Yield/
│   ├── image1.jpg
│   └── ...
└── [other_classes]/
```

### 2. Run Training Pipeline

```bash
# From project root
cd src

# Run with default dataset path (update in train.py if needed)
python train.py --config ../configs/traffic_signs.yaml

# Or specify custom dataset path
python train.py --data_path /path/to/LISATS --config ../configs/traffic_signs.yaml
```

### 3. View Results

Results are saved to `results/` and `models/` directories:
- `cv_results.csv` - Cross-validation results
- `test_results.csv` - Test set metrics
- `confusion_matrix_*.png` - Confusion matrices per model
- `roc_curves_*.png` - ROC curves
- `pr_curves_*.png` - Precision-Recall curves
- `calibration_*.png` - Calibration plots
- `model_comparison_*.png` - Model comparison charts
- `*.pkl` files in `models/` - Trained model pickles

## Pipeline Overview

### Step 1: Data Loading
- Load images from class directories
- Display dataset statistics
- Preserve original images for feature extraction

### Step 2: Stratified Split
- **Train**: 70% (stratified)
- **Validation**: 15% (stratified)
- **Test**: 15% (stratified)
- Maintains class distribution across splits

### Step 3: Feature Engineering

**HOG (Histogram of Oriented Gradients)**
- Edge direction histograms
- Robust to lighting changes
- Dimension: ~1764 features

**Color Histogram**
- BGR channel histograms
- 32 bins per channel
- Dimension: 96 features

**Bag of Visual Words**
- SIFT keypoint descriptors
- KMeans clustering (100 clusters)
- Histogram of cluster assignments
- Dimension: 100 features

**Total**: ~1960 concatenated features

### Step 4: Preprocessing
- StandardScaler normalization (fitted on train set)
- Applied to train, val, test sets

### Step 5: Model Training (5-Fold CV)

**Models evaluated**:
- **SVM** (RBF kernel, balanced)
- **Random Forest** (100 trees, balanced weights)
- **k-NN** (k=5, distance weighting)
- **XGBoost** (100 boosters, depth=6)

**Class Imbalance**: Addressed via class weights

### Step 6: Test Evaluation

**Metrics**:
- Accuracy (primary)
- Precision, Recall, F1-Score
- Brier Score (calibration)

**Visualizations**:
- Confusion matrices
- ROC curves (one-vs-rest)
- Precision-Recall curves
- Calibration plots
- Learning curves

## Configuration

Edit `configs/traffic_signs.yaml` to customize:

```yaml
features:
  hog:
    enabled: true
    cell_size: 8
  
  bovw:
    n_clusters: 100

training:
  n_splits: 5
  models:
    svm:
      C: 1.0
```

## Key Features

✅ **Reproducibility**
- Fixed random seeds
- Versioned datasets
- Logged hyperparameters

✅ **Robustness**
- Stratified sampling
- Class weight balancing
- Cross-validation

✅ **Comprehensive Evaluation**
- Multiple metrics
- Calibration analysis
- Feature importance
- Error analysis

✅ **Production-Ready**
- Modular code structure
- Configuration management
- Model persistence
- Comprehensive logging

## Usage Examples

### Load and Use Trained Model

```python
import pickle
import numpy as np
from src.features import FeatureExtractor

# Load trained model
with open('models/RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature extractor
fe = FeatureExtractor(n_clusters=100)
# Note: In practice, load the fitted extractor as well

# Extract features from new image
features = fe.extract_all_features([new_image])
features_scaled = fe.scale_features(features)

# Predict
prediction = model.predict(features_scaled)
probability = model.predict_proba(features_scaled)
```

### Analyze Results

```python
import pandas as pd

# Load results
cv_results = pd.read_csv('results/cv_results.csv')
test_results = pd.read_csv('results/test_results.csv')

# Best performing model
best_model = test_results.loc[test_results['Accuracy'].idxmax()]
print(f"Best Model: {best_model['Model']}")
print(f"Accuracy: {best_model['Accuracy']:.4f}")
```

## Experimental Notes

- **Dataset Size**: Scales to 1000+ images
- **Feature Extraction Time**: ~5 min for 1000 images
- **Training Time**: ~2 min with CV (5-fold)
- **Memory**: ~2GB for feature extraction and training

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

**Issue**: SIFT features not working
```bash
pip install opencv-contrib-python
```

**Issue**: Dataset path not found
- Update `--data_path` argument in `train.py`
- Check that LISATS directory has subdirectories for each class

## Next Steps

1. ✅ Complete training pipeline
2. ✅ Generate evaluation metrics
3. ✅ Create comprehensive report
4. 📋 Create presentation slides
5. 🚀 Deploy best model as API/service

## Contributing

For improvements or bug fixes:
1. Create feature branch
2. Make changes with comments
3. Test thoroughly
4. Submit pull request

## References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [OpenCV Documentation](https://docs.opencv.org/)

## Author

[Your Name]

## License

[Your License]

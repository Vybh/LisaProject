#!/bin/bash

# Traffic Sign Classification - Setup and Execution Guide
# This script helps you set up and run the complete ML pipeline

echo "=========================================="
echo "Traffic Sign Classification - Setup Guide"
echo "=========================================="

# Step 1: Environment Setup
echo -e "\n[1/5] Setting up Python environment..."
cd /Users/vybhavreddy/Desktop/tinylisaproject

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Step 2: Install Dependencies
echo -e "\n[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Step 3: Verify Dataset
echo -e "\n[3/5] Verifying dataset..."
DATASET_PATH="/Users/vybhavreddy/Desktop/LISATS"

if [ -d "$DATASET_PATH" ]; then
    echo "✓ Dataset found at $DATASET_PATH"
    echo "  Classes:"
    ls -1 "$DATASET_PATH" | head -5
    echo "  ..."
else
    echo "⚠️  Warning: Dataset not found at $DATASET_PATH"
    echo "  Please download LISATS dataset to this location"
fi

# Step 4: Run the Complete Pipeline
echo -e "\n[4/5] Running ML pipeline..."
echo "This will take 5-10 minutes depending on dataset size..."

cd src
python train.py --data_path "$DATASET_PATH" --config ../configs/traffic_signs.yaml

# Step 5: Run Jupyter Notebook for Analysis
echo -e "\n[5/5] Launch Jupyter notebook for detailed analysis..."
echo "To run the notebook:"
echo "  jupyter notebook ../notebooks/EDA.ipynb"
echo ""
echo "Or from the workspace root:"
echo "  python -m jupyter notebook notebooks/EDA.ipynb"

echo -e "\n=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Project Structure:"
echo "├── src/                          # Core ML modules"
echo "│   ├── datasets.py              # Data loading"
echo "│   ├── features.py              # Feature extraction"
echo "│   ├── models.py                # Model training"
echo "│   ├── evaluate.py              # Evaluation metrics"
echo "│   └── train.py                 # Main pipeline"
echo "├── configs/                      # Configuration files"
echo "│   └── traffic_signs.yaml       # ML hyperparameters"
echo "├── notebooks/                    # Jupyter notebooks"
echo "│   └── EDA.ipynb               # Complete analysis"
echo "├── models/                       # Trained model files"
echo "├── results/                      # Results & visualizations"
echo "└── requirements.txt              # Python dependencies"
echo ""
echo "Output files will be saved to results/ directory:"
echo "  - cv_results.csv                # Cross-validation metrics"
echo "  - test_results.csv              # Test set evaluation"
echo "  - confusion_matrix_*.png        # Confusion matrices"
echo "  - roc_curves_*.png              # ROC curves"
echo "  - pr_curves_*.png               # Precision-Recall curves"
echo "  - calibration_*.png             # Calibration plots"
echo "  - learning_curve_*.png          # Learning curves"
echo ""
echo "Trained models saved to models/ directory:"
echo "  - SVM.pkl"
echo "  - RandomForest.pkl"
echo "  - kNN.pkl"
echo "  - XGBoost.pkl"
echo ""

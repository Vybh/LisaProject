#!/usr/bin/env python3
"""
Traffic Sign Classification Project - File Navigator
Use this to understand the complete project structure and find what you need
"""

PROJECT_INDEX = {
    "GETTING STARTED": {
        "Quick Start": "QUICK_REFERENCE.md",
        "Full Setup Guide": "SETUP_GUIDE.md",
        "Run Automated Setup": "bash setup.sh",
        "Project Overview": "PROJECT_SUMMARY.md",
    },
    
    "MAIN EXECUTION": {
        "Run Pipeline (Command Line)": "cd src && python train.py",
        "Run Analysis (Interactive)": "jupyter notebook notebooks/EDA.ipynb",
        "Configuration": "configs/traffic_signs.yaml",
    },
    
    "SOURCE CODE (src/)": {
        "Data Loading": "src/datasets.py - TrafficSignDataset class",
        "Feature Extraction": "src/features.py - HOG, ColorHist, BoVW",
        "Model Training": "src/models.py - SVM, RF, k-NN, XGBoost",
        "Evaluation": "src/evaluate.py - Metrics, plots, visualizations",
        "Main Pipeline": "src/train.py - Orchestrate all steps",
    },
    
    "NOTEBOOKS": {
        "Complete Analysis": "notebooks/EDA.ipynb - 8 sections with all analysis",
    },
    
    "DOCUMENTATION": {
        "README": "README.md - Project overview and features",
        "Setup Guide": "SETUP_GUIDE.md - Installation and usage",
        "Report Template": "REPORT_TEMPLATE.md - 6-8 page research report",
        "Presentation Outline": "PRESENTATION_OUTLINE.md - 12-slide presentation",
        "Quick Reference": "QUICK_REFERENCE.md - Quick commands and tips",
        "This File": "INDEX.py - Project file navigator",
    },
    
    "OUTPUT FILES (Generated after running)": {
        "Trained Models": "models/*.pkl - SVM, RF, kNN, XGBoost pickles",
        "Results CSV": "results/cv_results.csv, results/test_results.csv",
        "Visualizations": "results/*.png - Confusion matrices, ROC, PR curves, etc.",
    },
    
    "PROJECT MODULES": {
        "Initialize Dataset": {
            "File": "src/datasets.py",
            "Class": "TrafficSignDataset",
            "Methods": [
                "load_data() - Load images from disk",
                "stratified_split() - 70/15/15 split",
                "get_class_weights() - Calculate imbalance weights",
            ]
        },
        
        "Extract Features": {
            "File": "src/features.py",
            "Class": "FeatureExtractor",
            "Methods": [
                "extract_hog() - Histogram of Oriented Gradients",
                "extract_color_histogram() - RGB color distribution",
                "fit_bovw() - Fit Bag of Visual Words",
                "extract_bovw() - Extract BoVW features",
                "extract_all_features() - Concatenate all 3 types",
                "fit_scaler() - Fit StandardScaler",
                "scale_features() - Normalize features",
            ]
        },
        
        "Train Models": {
            "File": "src/models.py",
            "Class": "ModelTrainer",
            "Methods": [
                "get_models() - Initialize 4 algorithms",
                "cross_validate_models() - 5-fold CV",
                "train_final_models() - Train on full train set",
                "evaluate_models() - Test set evaluation",
                "get_feature_importance() - Feature analysis",
            ]
        },
        
        "Evaluate & Visualize": {
            "File": "src/evaluate.py",
            "Class": "Evaluator",
            "Methods": [
                "plot_confusion_matrix() - Confusion heatmap",
                "plot_roc_curves() - ROC curves",
                "plot_precision_recall() - PR curves",
                "plot_calibration_curve() - Calibration analysis",
                "plot_model_comparison() - Model comparison bar charts",
                "plot_learning_curves() - Learning curve plots",
            ]
        },
    },
    
    "DATA FLOW": {
        "Step 1": "Load images → TrafficSignDataset.load_data()",
        "Step 2": "Stratified split → dataset.stratified_split()",
        "Step 3": "Extract features → FeatureExtractor.extract_all_features()",
        "Step 4": "Scale features → fe.fit_transform(), fe.scale_features()",
        "Step 5": "Cross-validate → ModelTrainer.cross_validate_models()",
        "Step 6": "Train final models → trainer.train_final_models()",
        "Step 7": "Evaluate → trainer.evaluate_models()",
        "Step 8": "Visualize → Evaluator methods",
        "Output": "models/*.pkl + results/*.csv + results/*.png",
    },
    
    "FEATURES EXPLANATION": {
        "HOG (Histogram of Oriented Gradients)": {
            "Dimensions": "1764 features",
            "Purpose": "Detect edges and corners",
            "Advantage": "Robust to lighting changes",
        },
        "Color Histogram": {
            "Dimensions": "96 features (32 bins × 3 channels)",
            "Purpose": "Capture color distribution",
            "Advantage": "Discriminates color-based signs",
        },
        "Bag of Visual Words (BoVW)": {
            "Dimensions": "100 features (100 KMeans clusters)",
            "Purpose": "Identify local visual patterns",
            "Advantage": "Captures complex local structures",
        },
        "Total": "1960 concatenated features",
    },
    
    "MODELS COMPARISON": {
        "SVM": {
            "Type": "Non-linear classifier",
            "Kernel": "RBF (Radial Basis Function)",
            "Advantage": "Effective in high-dimensional space",
            "Training Time": "Medium",
        },
        "Random Forest": {
            "Type": "Ensemble of decision trees",
            "Trees": "100 estimators",
            "Advantage": "Feature importance, robust",
            "Training Time": "Fast",
        },
        "k-Nearest Neighbors": {
            "Type": "Distance-based classifier",
            "k": "5 neighbors",
            "Advantage": "Simple, interpretable",
            "Training Time": "Negligible",
        },
        "XGBoost": {
            "Type": "Gradient boosting",
            "Estimators": "100 boosters",
            "Advantage": "High performance, handles class imbalance",
            "Training Time": "Medium",
        },
    },
    
    "EVALUATION METRICS": {
        "Primary": "Accuracy - Overall correctness",
        "Secondary": [
            "Precision - True positives / All positives",
            "Recall - True positives / All true instances",
            "F1-Score - Harmonic mean of precision & recall",
        ],
        "Calibration": "Brier Score - Prediction confidence reliability",
        "Visualizations": [
            "Confusion matrices - Misclassification patterns",
            "ROC curves - True positive vs false positive rate",
            "PR curves - Precision vs Recall trade-off",
            "Calibration plots - Probability calibration",
            "Learning curves - Bias-variance analysis",
        ],
    },
    
    "CONFIGURATION OPTIONS": {
        "File": "configs/traffic_signs.yaml",
        "Dataset": [
            "path: Dataset directory",
            "train_size: 0.7",
            "val_size: 0.15",
            "test_size: 0.15",
        ],
        "Features": [
            "hog.enabled: true/false",
            "hog.cell_size: 8 (default)",
            "color_histogram.bins: 32 (default)",
            "bovw.n_clusters: 100 (default)",
        ],
        "Training": [
            "n_splits: 5 (CV folds)",
            "random_state: 42 (reproducibility)",
            "Model hyperparameters: C, n_estimators, etc.",
        ],
    },
    
    "OUTPUT STRUCTURE": {
        "models/": {
            "SVM.pkl": "Trained SVM model",
            "RandomForest.pkl": "Trained Random Forest",
            "kNN.pkl": "Trained k-NN model",
            "XGBoost.pkl": "Trained XGBoost model",
        },
        "results/": {
            "cv_results.csv": "Cross-validation metrics",
            "test_results.csv": "Test set evaluation",
            "confusion_matrix_*.png": "4 confusion matrices",
            "roc_curves_*.png": "ROC curves per model",
            "pr_curves_*.png": "PR curves per model",
            "calibration_*.png": "Calibration plots",
            "learning_curve_*.png": "Learning curves",
            "model_comparison_*.png": "Comparison charts",
            "summary.json": "Final results summary",
        },
    },
    
    "COMMON WORKFLOWS": {
        "First Time Setup": [
            "1. bash setup.sh",
            "2. Wait for environment setup",
            "3. Dataset verification",
            "4. Automatic pipeline execution",
        ],
        
        "Manual Execution": [
            "1. source venv/bin/activate",
            "2. cd src",
            "3. python train.py --data_path /path/to/LISATS",
            "4. Check results/ and models/ folders",
        ],
        
        "Interactive Analysis": [
            "1. jupyter notebook notebooks/EDA.ipynb",
            "2. Run all cells sequentially",
            "3. Examine plots and results",
            "4. Modify code and re-run as needed",
        ],
        
        "Custom Configuration": [
            "1. Edit configs/traffic_signs.yaml",
            "2. Modify hyperparameters",
            "3. Save changes",
            "4. python train.py --config ../configs/traffic_signs.yaml",
        ],
    },
    
    "TROUBLESHOOTING": {
        "Module not found": "pip install -r requirements.txt",
        "Dataset not found": "Check /Users/vybhavreddy/Desktop/LISATS exists",
        "Out of memory": "Reduce dataset size or use subset",
        "SIFT not available": "pip install opencv-contrib-python",
        "Jupyter not found": "pip install jupyter",
        "Permission denied": "chmod +x setup.sh",
    },
    
    "NEXT STEPS": {
        "Immediate": [
            "☐ Run setup.sh",
            "☐ Execute pipeline",
            "☐ Verify output files",
        ],
        "This Week": [
            "☐ Open Jupyter notebook",
            "☐ Analyze results",
            "☐ Review visualizations",
        ],
        "Next 1-2 Weeks": [
            "☐ Write research report",
            "☐ Create presentation",
            "☐ Fine-tune best model",
        ],
        "Future": [
            "☐ Try deep learning",
            "☐ Add transfer learning",
            "☐ Deploy to production",
        ],
    },
}

def print_project_structure():
    """Print the complete project structure in an organized way."""
    print("\n" + "="*70)
    print("TRAFFIC SIGN CLASSIFICATION PROJECT - COMPLETE FILE GUIDE")
    print("="*70 + "\n")
    
    for section, content in PROJECT_INDEX.items():
        print(f"\n📁 {section}")
        print("-" * 70)
        
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    print(f"   {key}:")
                    for item in value:
                        print(f"     • {item}")
                else:
                    print(f"   • {key}: {value}")
        else:
            print(f"   {content}")

def print_quick_start():
    """Print quick start instructions."""
    print("\n" + "="*70)
    print("QUICK START (Choose one):")
    print("="*70)
    print("""
OPTION 1 - Automated (Recommended):
    bash setup.sh

OPTION 2 - Manual:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd src
    python train.py --data_path /Users/vybhavreddy/Desktop/LISATS

OPTION 3 - Interactive:
    jupyter notebook notebooks/EDA.ipynb
    """)

def print_file_purposes():
    """Print what each file does."""
    print("\n" + "="*70)
    print("FILE PURPOSES")
    print("="*70 + "\n")
    
    purposes = {
        "setup.sh": "Automated environment setup and pipeline execution",
        "README.md": "Main project documentation and overview",
        "QUICK_REFERENCE.md": "Quick commands, tips, and common tasks",
        "SETUP_GUIDE.md": "Detailed installation and usage instructions",
        "REPORT_TEMPLATE.md": "Structure for 6-8 page research report",
        "PRESENTATION_OUTLINE.md": "Structure for 10-12 slide presentation",
        "PROJECT_SUMMARY.md": "Project completion checklist and summary",
        "src/datasets.py": "Load and split image dataset",
        "src/features.py": "Extract HOG, ColorHist, BoVW features",
        "src/models.py": "Train SVM, RF, k-NN, XGBoost with CV",
        "src/evaluate.py": "Generate evaluation metrics and visualizations",
        "src/train.py": "Main ML pipeline orchestrator",
        "configs/traffic_signs.yaml": "Configuration for all hyperparameters",
        "notebooks/EDA.ipynb": "Interactive Jupyter analysis (8 sections)",
        "requirements.txt": "Python package dependencies",
    }
    
    for filename, purpose in purposes.items():
        print(f"  {filename:40} → {purpose}")

if __name__ == "__main__":
    import sys
    
    print("\n🚀 TRAFFIC SIGN CLASSIFICATION PROJECT - FILE NAVIGATOR\n")
    print("Usage: python INDEX.py [command]")
    print("Commands:")
    print("  structure  - Print complete project structure")
    print("  quickstart - Print quick start instructions")
    print("  purposes   - Print file purposes")
    print("  all        - Print all information (default)")
    
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if command in ["structure", "all"]:
        print_project_structure()
    
    if command in ["quickstart", "all"]:
        print_quick_start()
    
    if command in ["purposes", "all"]:
        print_file_purposes()
    
    print("\n" + "="*70)
    print("✅ Ready to start! Run: bash setup.sh")
    print("="*70 + "\n")

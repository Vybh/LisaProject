"""
Generate visualizations for all trained models.
This script loads saved models and generates plots.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import TrafficSignDataset
from features import FeatureExtractor
from evaluate import Evaluator
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--config', type=str, default='configs/traffic_signs.yaml', help='Config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading dataset...")
    dataset = TrafficSignDataset(args.data_path, random_state=config['random_state'])
    X, y = dataset.load_data()
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.stratified_split()
    
    print("Extracting features...")
    fe = FeatureExtractor(
        n_clusters=config['features']['bovw']['n_clusters'],
        scaler_type=config['preprocessing']['scaler_type']
    )
    
    print("Fitting BoVW on training set...")
    fe.fit_bovw(X_train)
    
    print("Extracting features from all sets...")
    X_train_features = fe.extract_all_features(X_train)
    X_val_features = fe.extract_all_features(X_val)
    X_test_features = fe.extract_all_features(X_test)
    
    print("Scaling features...")
    fe.fit_scaler(X_train_features)
    X_train_scaled = fe.scale_features(X_train_features)
    X_val_scaled = fe.scale_features(X_val_features)
    X_test_scaled = fe.scale_features(X_test_features)
    
    print("\nGenerating visualizations for all models...")
    evaluator = Evaluator(output_dir='results')
    
    model_names = ['SVM', 'RandomForest', 'kNN', 'XGBoost']
    
    for model_name in model_names:
        model_path = Path('models') / f'{model_name}.pkl'
        
        if not model_path.exists():
            print(f"Model {model_name}.pkl not found!")
            continue
        
        print(f"\n--- {model_name} ---")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        try:
            # Get predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
            
            # Confusion matrix
            print(f"  Generating confusion matrix...")
            evaluator.plot_confusion_matrix(y_test, y_pred, dataset.class_names, model_name)
            
            # ROC curves
            print(f"  Generating ROC curves...")
            evaluator.plot_roc_curves(y_test, y_proba, dataset.class_names, model_name)
            
            # Precision-Recall curves
            print(f"  Generating PR curves...")
            evaluator.plot_precision_recall(y_test, y_proba, dataset.class_names, model_name)
            
            # Calibration
            print(f"  Generating calibration curve...")
            y_proba_max = np.max(y_proba, axis=1)
            brier = evaluator.plot_calibration_curve(y_test, y_proba_max, model_name)
            print(f"  Brier Score: {brier:.4f}")
            
            print(f"✓ {model_name} plots generated successfully!")
            
        except Exception as e:
            print(f"✗ Error generating plots for {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nAll visualizations completed!")

if __name__ == '__main__':
    main()

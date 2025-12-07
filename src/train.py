"""
Main training script - orchestrates the full ML pipeline.
"""

import argparse
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from datasets import TrafficSignDataset
from features import FeatureExtractor
from models import ModelTrainer
from evaluate import Evaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: dict, output_path: str) -> None:
    """Save results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results_serializable = convert_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Traffic Sign Classification')
    parser.add_argument('--config', type=str, default='configs/traffic_signs.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default='/Users/vybhavreddy/Desktop/LISATS',
                       help='Path to dataset')
    args = parser.parse_args()
    
    print("="*60)
    print("Traffic Sign Classification Pipeline")
    print("="*60)
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"\nLoaded config from {args.config}")
    else:
        # Use default config
        config = {
            'n_clusters': 100,
            'scaler_type': 'standard',
            'random_state': 42,
            'n_splits': 5
        }
        print("\nUsing default configuration")
    
    # Create output directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Initialize dataset
    print("\n" + "="*60)
    print("STEP 1: Load Data")
    print("="*60)
    
    dataset = TrafficSignDataset(args.data_path, random_state=config['random_state'])
    X, y = dataset.load_data()
    
    # Stratified split
    print("\n" + "="*60)
    print("STEP 2: Stratified Data Split")
    print("="*60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.stratified_split()
    
    # Initialize feature extractor and fit BoVW on training set
    print("\n" + "="*60)
    print("STEP 3: Feature Extraction & Engineering")
    print("="*60)
    
    fe = FeatureExtractor(
        n_clusters=config['features']['bovw']['n_clusters'],
        scaler_type=config['preprocessing']['scaler_type']
    )
    
    print("\nFitting BoVW on training set...")
    fe.fit_bovw(X_train)
    
    print("\nExtracting features from training set...")
    X_train_features = fe.extract_all_features(X_train)
    
    print("\nExtracting features from validation set...")
    X_val_features = fe.extract_all_features(X_val)
    
    print("\nExtracting features from test set...")
    X_test_features = fe.extract_all_features(X_test)
    
    # Scale features
    print("\nScaling features...")
    X_train_scaled = fe.fit_transform(X_train_features)
    X_val_scaled = fe.scale_features(X_val_features)
    X_test_scaled = fe.scale_features(X_test_features)
    
    # Cross-validation on training set
    print("\n" + "="*60)
    print("STEP 4: Cross-Validation")
    print("="*60)
    
    trainer = ModelTrainer(
        random_state=config['training']['random_state'],
        n_splits=config['training']['n_splits']
    )
    
    class_weights = dataset.get_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")
    
    cv_results = trainer.cross_validate_models(X_train_scaled, y_train, class_weights)
    print("\n" + cv_results.to_string(index=False))
    
    # Save CV results
    cv_results.to_csv('results/cv_results.csv', index=False)
    print("\nSaved: results/cv_results.csv")
    
    # Train final models
    print("\n" + "="*60)
    print("STEP 5: Train Final Models")
    print("="*60)
    
    trainer.train_final_models(X_train_scaled, y_train, class_weights)
    
    # Save trained models
    for model_name, model in trainer.best_models.items():
        model_path = Path(f'models/{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved: {model_path}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("STEP 6: Evaluation on Test Set")
    print("="*60)
    
    test_results = trainer.evaluate_models(X_test_scaled, y_test)
    test_results.to_csv('results/test_results.csv', index=False)
    print("\nSaved: results/test_results.csv")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("STEP 7: Generate Visualizations")
    print("="*60)
    
    evaluator = Evaluator(output_dir='results')
    
    for model_name, model in trainer.best_models.items():
        print(f"\nGenerating visualizations for {model_name}...")
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Confusion matrix
        evaluator.plot_confusion_matrix(y_test, y_pred, dataset.class_names, model_name)
        
        # ROC curves
        evaluator.plot_roc_curves(y_test, y_proba, dataset.class_names, model_name)
        
        # Precision-Recall curves
        evaluator.plot_precision_recall(y_test, y_proba, dataset.class_names, model_name)
        
        # Calibration
        y_proba_max = np.max(y_proba, axis=1)
        brier = evaluator.plot_calibration_curve(y_test, y_proba_max, model_name)
        
        # Store Brier score
        test_results.loc[test_results['Model'] == model_name, 'Brier_Score'] = brier
    
    # Model comparison plots
    evaluator.plot_model_comparison(test_results, 'Accuracy')
    
    # Save updated results with calibration
    test_results.to_csv('results/test_results_with_calibration.csv', index=False)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to: results/")
    print(f"Models saved to: models/")


if __name__ == '__main__':
    main()

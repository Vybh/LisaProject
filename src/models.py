"""
Machine learning models and training utilities.
Implements SVM, Random Forest, k-NN, and XGBoost.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from typing import Dict, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and evaluate multiple ML models with cross-validation."""
    
    def __init__(self, random_state: int = 42, n_splits: int = 5):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
            n_splits: Number of CV folds
        """
        self.random_state = random_state
        self.n_splits = n_splits
        self.models = {}
        self.cv_results = {}
        self.best_models = {}
        
    def get_models(self, class_weights: Dict = None) -> Dict[str, Any]:
        """
        Create model instances with hyperparameters.
        
        Args:
            class_weights: Dictionary of class weights for imbalance handling
            
        Returns:
            Dictionary of model name -> model instance
        """
        models = {
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced' if class_weights else None,
                random_state=self.random_state,
                probability=True
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced' if class_weights else None,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'kNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=None,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        }
        
        self.models = models
        return models
    
    def cross_validate_models(self, X: np.ndarray, y: np.ndarray, 
                              class_weights: Dict = None) -> pd.DataFrame:
        """
        Train models with 5-fold stratified cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            class_weights: Class weights dictionary
            
        Returns:
            DataFrame with CV results
        """
        models = self.get_models(class_weights)
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                            random_state=self.random_state)
        
        results_list = []
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Cross-validation
            cv_scores = cross_validate(
                model, X, y, cv=cv,
                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                return_train_score=True,
                n_jobs=1
            )
            
            # Store results
            result = {
                'Model': model_name,
                'Accuracy_Mean': cv_scores['test_accuracy'].mean(),
                'Accuracy_Std': cv_scores['test_accuracy'].std(),
                'Precision_Mean': cv_scores['test_precision_macro'].mean(),
                'Recall_Mean': cv_scores['test_recall_macro'].mean(),
                'F1_Mean': cv_scores['test_f1_macro'].mean(),
                'Train_Acc_Mean': cv_scores['train_accuracy'].mean(),
            }
            
            results_list.append(result)
            self.cv_results[model_name] = cv_scores
            
            print(f"  Acc: {result['Accuracy_Mean']:.4f} (±{result['Accuracy_Std']:.4f})")
            print(f"  F1:  {result['F1_Mean']:.4f}")
        
        results_df = pd.DataFrame(results_list)
        return results_df
    
    def train_final_models(self, X_train: np.ndarray, y_train: np.ndarray,
                          class_weights: Dict = None) -> Dict[str, Any]:
        """
        Train final models on full training set.
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weights: Class weights dictionary
            
        Returns:
            Dictionary of trained model instances
        """
        models = self.get_models(class_weights)
        
        print("\n=== Training Final Models ===")
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)
            self.best_models[model_name] = model
        
        return self.best_models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate trained models on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with test results
        """
        if not self.best_models:
            raise ValueError("No trained models found. Call train_final_models() first.")
        
        results_list = []
        
        print("\n=== Test Set Performance ===")
        for model_name, model in self.best_models.items():
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            result = {
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1_Score': f1,
            }
            
            results_list.append(result)
            
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
        
        results_df = pd.DataFrame(results_list)
        return results_df
    
    def get_feature_importance(self, model_name: str, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of model (RF or XGBoost)
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not trained.")
        
        model = self.best_models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {model_name} does not have feature_importances_")
        
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df

"""
Evaluation and analysis module.
Generates confusion matrices, ROC curves, PR curves, and calibration metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from pathlib import Path
from typing import Dict, List


class Evaluator:
    """Generate comprehensive evaluation metrics and visualizations."""
    
    def __init__(self, output_dir: str = 'results'):
        """Initialize evaluator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str], model_name: str) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            model_name: Name of model
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.output_dir / f'confusion_matrix_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                       class_names: List[str], model_name: str) -> Dict[str, float]:
        """
        Plot ROC curves for one-vs-rest.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities (n_samples, n_classes)
            class_names: List of class names
            model_name: Name of model
            
        Returns:
            Dictionary of AUC scores
        """
        n_classes = len(class_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        auc_scores = {}
        
        for i in range(n_classes):
            # One-vs-rest
            y_binary = (y_true == i).astype(int)
            
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[class_names[i]] = roc_auc
            
            ax = axes[i % 4]
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC - {class_names[i]} vs Rest')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'ROC Curves - {model_name}', fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / f'roc_curves_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        return auc_scores
    
    def plot_precision_recall(self, y_true: np.ndarray, y_proba: np.ndarray,
                             class_names: List[str], model_name: str) -> None:
        """
        Plot precision-recall curves.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_names: List of class names
            model_name: Name of model
        """
        n_classes = len(class_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for i in range(n_classes):
            y_binary = (y_true == i).astype(int)
            
            precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, i])
            pr_auc = auc(recall, precision)
            
            ax = axes[i % 4]
            ax.plot(recall, precision, color='green', lw=2,
                   label=f'AUC = {pr_auc:.3f}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'PR Curve - {class_names[i]}')
            ax.legend(loc="lower left")
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
        
        plt.suptitle(f'Precision-Recall Curves - {model_name}', fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / f'pr_curves_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                              model_name: str) -> float:
        """
        Plot calibration curve and return Brier score.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities (max over classes for binary)
            model_name: Name of model
            
        Returns:
            Brier score
        """
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        brier = brier_score_loss(y_true, y_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name)
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Plot - {model_name} (Brier: {brier:.4f})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / f'calibration_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        return brier
    
    def plot_model_comparison(self, results_df: pd.DataFrame, metric: str = 'Accuracy') -> None:
        """
        Plot comparison of models across metric.
        
        Args:
            results_df: DataFrame with model results
            metric: Metric to compare
        """
        plt.figure(figsize=(10, 6))
        
        models = results_df['Model']
        values = results_df[metric]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(models, values, color=colors[:len(models)], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.ylabel(metric)
        plt.title(f'Model Comparison - {metric}')
        plt.ylim([0, max(values) * 1.1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        safe_metric = metric.replace('/', '_').replace(' ', '_')
        save_path = self.output_dir / f'model_comparison_{safe_metric}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_learning_curves(self, train_sizes: np.ndarray, train_scores: np.ndarray,
                            val_scores: np.ndarray, model_name: str) -> None:
        """
        Plot learning curves showing train/val performance vs dataset size.
        
        Args:
            train_sizes: Array of training set sizes
            train_scores: Array of training scores
            val_scores: Array of validation scores
            model_name: Name of model
        """
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                         alpha=0.2, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score', linewidth=2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                         alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / f'learning_curve_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

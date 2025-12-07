"""
Dataset loading and preparation module.
Handles data splits using stratified 70/15/15 split.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from typing import Tuple, List, Dict


class TrafficSignDataset:
    """Load and manage traffic sign dataset."""
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialize dataset loader.
        
        Args:
            data_path: Path to LISATS dataset directory
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.X = None
        self.y = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all images and labels from dataset.
        Supports both class-folder structure and flat CSV-annotated structure.
        Images are resized to 64x64 for consistent processing.
        
        Returns:
            Tuple of (images array, labels array)
        """
        images = []
        labels = []
        IMG_SIZE = 64  # Resize all images to 64x64
        
        # Check if annotations.csv exists (flat structure)
        annotations_file = self.data_path / 'annotations.csv'
        
        if annotations_file.exists():
            # Load from CSV annotations
            df = pd.read_csv(annotations_file)
            
            # Get unique classes and encode them
            unique_classes = sorted(df['class'].unique())
            self.class_names = unique_classes
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            
            print(f"Found annotations.csv with {len(df)} images")
            
            for idx, row in df.iterrows():
                img_path = self.data_path / row['filename']
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Resize to consistent size
                        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img_resized)
                        labels.append(class_to_idx[row['class']])
            
            self.X = np.array(images)
            self.y = np.array(labels)
            
        else:
            # Load from class folders (original structure)
            for class_idx, class_dir in enumerate(sorted(self.data_path.iterdir())):
                if not class_dir.is_dir():
                    continue
                    
                self.class_names.append(class_dir.name)
                
                # Load images from class directory
                for img_path in sorted(class_dir.glob('*.jpg')) + sorted(class_dir.glob('*.png')):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Resize to consistent size
                        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img_resized)
                        labels.append(class_idx)
            
            self.X = np.array(images)
            self.y = np.array(labels)
        
        print(f"Loaded {len(images)} images from {len(self.class_names)} classes")
        print(f"Class names: {self.class_names}")
        print(f"Dataset shape: {self.X.shape}")
        print(f"Labels distribution:\n{pd.Series(self.y).value_counts().sort_index()}")
        
        return self.X, self.y
    
    def stratified_split(self, test_size: float = 0.3, val_ratio: float = 0.5) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train (70%), validation (15%), and test (15%) using stratified split.
        
        Args:
            test_size: Combined test+val size (0.3 = 30%)
            val_ratio: Ratio of val to test (0.5 = split remaining 30% equally)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # First split: 70% train, 30% temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            stratify=self.y,
            random_state=self.random_state
        )
        
        # Second split: Split temp 50/50 into val and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        print(f"\n=== Data Split (Stratified 70/15/15) ===")
        print(f"Train: {len(X_train)} samples ({len(X_train)/len(self.X)*100:.1f}%)")
        print(f"Val:   {len(X_val)} samples ({len(X_val)/len(self.X)*100:.1f}%)")
        print(f"Test:  {len(X_test)} samples ({len(X_test)/len(self.X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights to handle imbalance (for training).
        
        Args:
            y: Label array
            
        Returns:
            Dictionary mapping class idx to weight
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        weights = {}
        for cls, count in zip(unique, counts):
            weights[cls] = total / (len(unique) * count)
        return weights

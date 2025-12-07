"""
Feature extraction module.
Implements HOG, ColorHist, and BoVW features.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Extract image features: HOG, Color Histogram, Bag of Visual Words."""
    
    def __init__(self, n_clusters: int = 100, scaler_type: str = 'standard'):
        """
        Initialize feature extractor.
        
        Args:
            n_clusters: Number of clusters for BoVW
            scaler_type: 'standard' or 'minmax'
        """
        self.n_clusters = n_clusters
        self.scaler_type = scaler_type
        self.kmeans = None
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.is_fitted = False
        
    def extract_hog(self, image: np.ndarray, cell_size: int = 8, 
                    block_size: int = 2) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients (HOG).
        
        Args:
            image: BGR image
            cell_size: Size of cell (default 8x8)
            block_size: Size of block (default 2x2 cells)
            
        Returns:
            HOG feature vector
        """
        # Use 64x64 image directly (already resized in load_data)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute HOG with parameters compatible with 64x64 images
        # winSize must be a multiple of blockStride
        hog = cv2.HOGDescriptor(
            (64, 64),    # winSize - must match image size
            (16, 16),    # blockSize
            (8, 8),      # blockStride
            (8, 8),      # cellSize
            9            # nbins (number of orientation bins)
        )
        
        hog_features = hog.compute(gray)
        if hog_features.size == 0:
            # Fallback: compute manually if cv2.HOG fails
            return np.zeros(324)  # Default HOG feature size for 64x64
        
        return hog_features.flatten()
    
    def extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Extract color histogram features.
        
        Args:
            image: BGR image
            bins: Number of histogram bins per channel
            
        Returns:
            Color histogram feature vector
        """
        # Resize for consistency
        image = cv2.resize(image, (128, 128))
        
        # Split channels and compute histograms
        features = []
        for i in range(3):  # B, G, R
            hist = cv2.calcHist(
                [image], [i], None, [bins], [0, 256]
            )
            features.append(hist.flatten())
        
        return np.concatenate(features)
    
    def extract_sift_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Extract SIFT keypoints for BoVW.
        
        Args:
            image: BGR image
            
        Returns:
            Array of SIFT descriptors
        """
        # Resize image
        image = cv2.resize(image, (128, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            # Return dummy descriptors if none found
            return np.zeros((1, 128))
        
        return descriptors
    
    def fit_bovw(self, images: List[np.ndarray]) -> None:
        """
        Fit Bag of Visual Words model (KMeans clustering).
        
        Args:
            images: List of images to fit on (typically training set)
        """
        print(f"Extracting SIFT descriptors from {len(images)} images...")
        all_descriptors = []
        
        for i, img in enumerate(images):
            if i % max(1, len(images) // 10) == 0:
                print(f"  Processing image {i+1}/{len(images)}")
            descriptors = self.extract_sift_keypoints(img)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        # Concatenate all descriptors
        all_descriptors = np.vstack(all_descriptors)
        
        print(f"Clustering {len(all_descriptors)} descriptors into {self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=3, max_iter=100)
        self.kmeans.fit(all_descriptors)
        print("BoVW model fitted.")
    
    def extract_bovw(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Bag of Visual Words features.
        
        Args:
            image: BGR image
            
        Returns:
            BoVW histogram feature vector
        """
        if self.kmeans is None:
            raise ValueError("BoVW model not fitted. Call fit_bovw() first.")
        
        descriptors = self.extract_sift_keypoints(image)
        
        # Assign descriptors to nearest cluster
        labels = self.kmeans.predict(descriptors)
        
        # Create histogram
        histogram = np.bincount(labels, minlength=self.n_clusters)
        
        # Normalize
        histogram = histogram / (np.sum(histogram) + 1e-6)
        
        return histogram
    
    def extract_all_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract all features for multiple images.
        Concatenates HOG + ColorHist + BoVW.
        
        Args:
            images: List of images
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features_list = []
        
        for i, img in enumerate(images):
            if i % max(1, len(images) // 10) == 0:
                print(f"Extracting features from image {i+1}/{len(images)}")
            
            hog = self.extract_hog(img)
            color_hist = self.extract_color_histogram(img)
            bovw = self.extract_bovw(img)
            
            # Concatenate all features
            all_features = np.concatenate([hog, color_hist, bovw])
            features_list.append(all_features)
        
        X = np.array(features_list)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"  HOG features: {len(hog)}")
        print(f"  ColorHist features: {len(color_hist)}")
        print(f"  BoVW features: {len(bovw)}")
        
        return X
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit feature scaler on training data."""
        self.scaler.fit(X)
        self.is_fitted = True
    
    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using fitted scaler.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform in one step."""
        self.fit_scaler(X)
        return self.scale_features(X)

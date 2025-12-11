"""
Extractor combinado de características de Color (HSV Histogram) y Textura (LBP).
"""

import numpy as np
from typing import Union
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern

from .base_extractor import BaseExtractor


class ColorTextureExtractor(BaseExtractor):
    """
    Combina características de color (histograma HSV) y textura (LBP).
    """
    
    def __init__(self, hsv_bins=(8, 8, 8), lbp_points=24, lbp_radius=3):
        """
        Inicializa el extractor.
        
        Args:
            hsv_bins: Número de bins para cada canal HSV
            lbp_points: Número de puntos en el patrón circular de LBP
            lbp_radius: Radio del patrón circular de LBP
        """
        super().__init__("ColorTexture")
        self.hsv_bins = hsv_bins
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        
        # Dimensión: HSV histogram + LBP histogram
        self.hsv_dim = hsv_bins[0] * hsv_bins[1] * hsv_bins[2]
        self.lbp_dim = lbp_points + 2
        self.feature_dim = self.hsv_dim + self.lbp_dim
        
    def extract_color_features(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extrae histograma de color en espacio HSV.
        
        Args:
            img_bgr: Imagen en formato BGR
            
        Returns:
            np.ndarray: Histograma HSV normalizado
        """
        # Convertir a HSV
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Calcular histograma 3D
        hist = cv2.calcHist(
            [img_hsv], 
            [0, 1, 2], 
            None, 
            self.hsv_bins, 
            [0, 180, 0, 256, 0, 256]
        )
        
        # Normalizar
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        
        return hist
    
    def extract_texture_features(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Extrae características de textura usando LBP.
        
        Args:
            img_gray: Imagen en escala de grises
            
        Returns:
            np.ndarray: Histograma LBP normalizado
        """
        # Calcular LBP
        lbp = local_binary_pattern(
            img_gray, 
            self.lbp_points, 
            self.lbp_radius, 
            method='uniform'
        )
        
        # Calcular histograma
        hist, _ = np.histogram(
            lbp.ravel(), 
            bins=self.lbp_points + 2, 
            range=(0, self.lbp_points + 2)
        )
        
        # Normalizar
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-7)
        
        return hist
    
    def extract(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extrae características combinadas de color y textura.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            np.ndarray: Vector de características normalizado
        """
        # Cargar imagen
        img = self.load_image(image)
        
        # Convertir a BGR para OpenCV
        if len(img.shape) == 2:  # Escala de grises
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convertir a escala de grises para LBP
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Extraer características
        color_features = self.extract_color_features(img_bgr)
        texture_features = self.extract_texture_features(img_gray)
        
        # Concatenar
        features = np.concatenate([color_features, texture_features])
        
        # Normalizar vector completo
        features = self.normalize_features(features)
        
        return features
    
    def get_feature_dim(self) -> int:
        return self.feature_dim

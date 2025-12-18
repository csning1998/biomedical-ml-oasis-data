
"""
This file is used to define the processors for RQ2.
It contains experimental techniques (CLAHE, Viridis), completely independent of RQ1's data_loader.
"""
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.resnet import preprocess_input


def apply_viridis(image_path):
    """
    Convert grayscale image to Viridis Colormap.
    The report states that Viridis is perceptually uniform and does not have the artificial edge artifacts of Jet.
    """
    if isinstance(image_path, bytes):
        image_path = image_path.decode('utf-8')

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Image not found: {image_path}")

    img_resized = cv2.resize(img_gray, (224, 224))

    # Apply Viridis
    img_color = cv2.applyColorMap(img_resized, cv2.COLORMAP_VIRIDIS)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    return img_rgb

def apply_clahe_jet(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    First apply CLAHE to enhance local contrast, then apply Jet.
    This is a combination suspected to 'amplify background noise leading to data leakage' in the report.
    """
    if isinstance(image_path, bytes):
        image_path = image_path.decode('utf-8')

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Image not found: {image_path}")

    img_resized = cv2.resize(img_gray, (224, 224))

    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_resized)

    # 2. Jet Colormap
    img_color = cv2.applyColorMap(img_clahe, cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    return img_rgb


def processor_viridis(file_path_tensor, label_tensor):
    """
    Inject the Viridis processor to create_dataset.
    """
    path = file_path_tensor.numpy()
    label = label_tensor.numpy()
    
    img = apply_viridis(path)
    
    # Standard ResNet preprocessing
    img = img.astype(np.float32)
    img = preprocess_input(img)
    
    return img, label

def processor_clahe_jet(file_path_tensor, label_tensor):
    """
    Inject the CLAHE+Jet processor to create_dataset.
    Use default parameter clip_limit=2.0.
    """
    path = file_path_tensor.numpy()
    label = label_tensor.numpy()
    
    img = apply_clahe_jet(path)
    
    img = img.astype(np.float32)
    img = preprocess_input(img)
    
    return img, label

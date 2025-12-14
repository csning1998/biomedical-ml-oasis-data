
"""
This file is used to load the data and apply the Pseudo-RGB transformation.
"""

import tensorflow as tf
import numpy as np
import cv2
import functools
from sklearn.utils import class_weight
from tensorflow.keras.applications.resnet import preprocess_input

# Define the CLASS_MAP at the module level, which is used to map the class name to the label.
CLASS_MAP = {
    "non-demented": 0, 
    "dementia_very_mild": 1, 
    "dementia_mild": 2, 
    "dementia_moderate": 3
}

def get_class_weights(df):
    """
    Computes class weights for imbalanced datasets based on the input DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'class_name' column.
        
    Returns:
        dict: A dictionary mapping class indices (int) to weights (float),
                ready to be passed to model.fit(class_weight=...).
    """
    # 1. Get class indices
    y_train_indices = [CLASS_MAP[c] for c in df['class_name'].values]
    
    # 2. Compute class weights
    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    
    # 3. Convert to Keras dictionary format {0: w0, 1: w1, ...}
    class_weights_dict = dict(enumerate(class_weights_values))
    
    return class_weights_dict

def _apply_enhancement(img_gray, method_str):
    """
    Internal helper: Applies pixel-level enhancement before colormapping.
    Handles 'clahe' and 'he' (Histogram Equalization).
    """
    if 'clahe' in method_str: # CLAHE: Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_gray = clahe.apply(img_gray)
    elif 'he' in method_str: # HE: Global Histogram Equalization
        img_gray = cv2.equalizeHist(img_gray)
    return img_gray


def apply_pseudo_rgb(image_path, method='duplicate'):
    """
    Reads a grayscale image and applies Enhancement + Pseudo-RGB transformation.
    
    Args:
        image_path (str): Path to the image file.
        method (str): A composite string indicating the pipeline.
                    Format: "[enhancement_][colormap]"
                    Examples: 
                        - 'duplicate' (Baseline)
                        - 'jet', 'viridis' (Standard)
                        - 'clahe_duplicate', 'he_jet' (Enhanced)
                        - 'sobel' (Edge Detection)
    Returns:
        np.array: A 3-channel RGB image with shape (224, 224, 3), range [0, 255].
    """
    # 1. Read as grayscale
    if isinstance(image_path, bytes):
        image_path = image_path.decode('utf-8')

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise ValueError(f"Image not found: {image_path}")

    # 2. Resize to 224x224 for Standard Input for ResNet-101
    img_resized = cv2.resize(img_gray, (224, 224))

    # 3. Enhancement Phase (Pre-processing)
    image_enhanced = _apply_enhancement(img_resized, method)

    # 4. Transformation Phase (Grayscale -> RGB)
    base_method = method
    for prefix in ['clahe_', 'he_']:
        if prefix in method:
            base_method = method.replace(prefix, '')

    if base_method == 'duplicate':
        # Stack the grayscale image 3 times: (224, 224) -> (224, 224, 3)
        img_rgb = np.stack([img_resized] * 3, axis=-1)
        
    elif base_method == 'jet':
        # Apply JET colormap: Maps low intensity to Blue, high to Red
        # applyColorMap returns BGR, thus convert to RGB
        img_color = cv2.applyColorMap(img_resized, cv2.COLORMAP_JET)
        img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        
    elif base_method == 'viridis':
        # Apply VIRIDIS colormap: Maps low intensity to Blue, high to Red
        try:
            img_color = cv2.applyColorMap(img_resized, cv2.COLORMAP_VIRIDIS)
        except AttributeError:
            # Fallback if specific version misses VIRIDIS
            img_color = cv2.applyColorMap(img_resized, cv2.COLORMAP_JET)
        img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    elif base_method == 'sobel':
        # Ablation Study: Sobel vs Texture
        sobelx = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude and Normalize to 0-255
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        img_rgb = np.stack([magnitude]*3, axis=-1)

    else:
        raise ValueError(f"Unknown method: {method}")
        
    return img_rgb


def image_processor(file_path_tensor, label_tensor, method):
    """
    Wrapper for tf.py_function. The Model Factory must now handle 'preprocess_input' via a Lambda layer.
    Returns raw RGB values [0, 255] in float32.
    """
    
    # 1. Convert tensors to numpy arrays
    file_path = file_path_tensor.numpy()
    label = label_tensor.numpy()
    
    # 2. Get [0, 255] RGB image
    img = apply_pseudo_rgb(file_path, method=method)
    
    # 3. Cast to float32 for Tensor compatibility
    img = img.astype(np.float32)

    # 4. Caffe Style Preprocessing
    img = preprocess_input(img)
    
    return img, label


def set_tensor_shapes(img, label):
    """
    Restores shape information lost during tf.py_function wrapping.
    
    Args:
        img (Tensor): Image tensor with unknown shape.
        label (Tensor): Label tensor with unknown shape.
        
    Returns:
        (Tensor, Tensor): Tensors with fixed shapes ((224, 224, 3), ()).
    """
    img.set_shape([224, 224, 3])  # To match ResNet-101 input requirements
    label.set_shape([])           # Label is a scalar (single integer)
    
    return img, label


def create_dataset(df, method='duplicate', batch_size=32, shuffle=False):
    """
    Factory function to build the tf.data pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        method (str): 'duplicate' or 'jet'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle data.
        
    Returns:
        tf.data.Dataset: Configured dataset pipeline.
    """
    # 1. Extract Data
    file_paths = df['file_path'].values
    labels = [CLASS_MAP[c] for c in df['class_name'].values]
    
    # 2. Create Base TensorSliceDataset
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # 3. Shuffle (Only for training)
    if shuffle:
        ds = ds.shuffle(buffer_size=2000, seed=42)
    
    # 4. Partial binding of the method argument
    loader_func = functools.partial(image_processor, method=method)
    
    # 5. Map: Connect Python logic to TF Graph
    ds = ds.map(
        lambda x, y: tf.py_function(
            func=loader_func,
            inp=[x, y],
            Tout=[tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # 6. Map: Fix Shapes
    ds = ds.map(set_tensor_shapes, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 7. Batch & Prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds


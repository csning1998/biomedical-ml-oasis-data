
"""
This file is used to load the data and create the dataset.
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

import tensorflow as tf

def build_baseline_model(input_shape=(224, 224, 3), num_classes=4, weights='imagenet'):
    """
    Constructs the ResNet-101 model for AD classification.
    
    Args:
        input_shape (tuple): The shape of input images.
        num_classes (int): Number of target classes (4 for OASIS).
        weights (str): 'imagenet' for transfer learning or None for random init.
        
    Returns:
        tf.keras.Model: The compiled model architecture.
    """
    # 1. 
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.ResNet101(
        include_top=False,
        weights=weights,  # 2.
        input_tensor=inputs
    )
    
    # 3. Feature Aggregation such that (Batch, 7, 7, 2048) -> (Batch, 2048)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    
    # 4. Classification Head such that (Batch, 2048) -> (Batch, 4)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # 5.  
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet101_AD_Classifier")

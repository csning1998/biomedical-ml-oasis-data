
"""
This file is used to build the model for RQ2 only
"""

import tensorflow as tf

"""
For all function:
freeze_bn (bool): Whether to freeze the batch normalization layers.
This parameter determines how the model handles the standardization statistics of features during fine-tuning.            
    * If True (Recommended for Small Batch Size): 
        Locks the BN layers' gamma (Scale) and beta (Shift) params, and forces the use of ImageNet pre-trained population statistics (mu_{pop}, sigma^2_{pop}).
        **Benefit**: Prevents "Noisy Gradients" caused by unstable batch statistics when batch size is small (e.g., 16), ensuring training stability.
            
    * If False (Default in pure Keras): 
        Allows BN layers to update gamma and beta via backpropagation, and uses the current batch statistics (bar{x}_{batch}, s^2_{batch}) for standardization.
        **Risk**: With small batches, sample statistics are poor estimators of population statistics, leading to biased gradients and potential model divergence.
"""

def build_resnet101(input_shape=(224, 224, 3), num_classes=4, freeze_bn=False, weights='imagenet'):
    """
    Build a ResNet101 model for RQ2 that is consistent with RQ1, but parameterize freeze_bn for control.

    Args:
        input_shape (tuple): The shape of the input image.
        num_classes (int): The number of classes.
        weights (str): The weights to use for the model.
    Returns:
        tf.keras.Model: The built model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    base_model = tf.keras.applications.ResNet101(
        include_top=False, 
        weights=weights, 
        input_tensor=inputs
    )
    
    # Ensure Backbone is tranable for fine-tuning
    base_model.trainable = True

    # Implement RQ1 validated BN Freezing strategy (to avoid Noisy Gradients, aka biased estimation of gradient.)
    if freeze_bn:
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    else:
        base_model.trainable = True

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet101_AD_Classifier")

def build_resnet50(input_shape=(224, 224, 3), num_classes=4, freeze_bn=False, weights='imagenet'):
    """
    Build a ResNet50 model, strictly following the architectural pattern of build_resnet101.
    Used for comparative analysis (Depth impact study).
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    base_model = tf.keras.applications.ResNet50(
        include_top=False, 
        weights=weights, 
        input_tensor=inputs
    )
    
    # SRP: Reusing the validated BN Freezing logic
    if freeze_bn:
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    else:
        base_model.trainable = True

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_AD_Classifier")


def build_efficientnet(subtype='B0', input_shape=(224, 224, 3), num_classes=4, freeze_bn=False, weights='imagenet'):
    """
    Build an EfficientNet model with switchable backbones (B0-B3).
    
    Args:
        subtype (str): The variant of EfficientNet to use. 
            Supported: 'B0', 'B1', 'B2', 'B3', 'V2B0', 'V2B1', 'V2B2', 'V2B3'.
        input_shape (tuple): The shape of the input image.
        num_classes (int): The number of classes.
        weights (str): Pretrained weights path or 'imagenet'.
        
    Returns:
        tf.keras.Model: The constructed EfficientNet model.
    """
    # 1. Switch Case Dispatcher (Dictionary Mapping)
    model_map = {
        'B0': tf.keras.applications.EfficientNetB0,
        'B1': tf.keras.applications.EfficientNetB1,
        'B2': tf.keras.applications.EfficientNetB2,
        'B3': tf.keras.applications.EfficientNetB3,
        'V2B0': tf.keras.applications.EfficientNetV2B0,
        'V2B1': tf.keras.applications.EfficientNetV2B1,
        'V2B2': tf.keras.applications.EfficientNetV2B2,
        'V2B3': tf.keras.applications.EfficientNetV2B3,
    }
    
    if subtype not in model_map:
        raise ValueError(f"Invalid EfficientNet subtype: {subtype}. Available: {list(model_map.keys())}")
    
    # 2. Instantiate the specific backbone
    backbone_constructor = model_map[subtype]
    inputs = tf.keras.Input(shape=input_shape)
    
    base_model = backbone_constructor(
        include_top=False, 
        weights=weights, 
        input_tensor=inputs
    )
    
    # 3. Apply BN Freezing Strategy
    if freeze_bn:
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    else:
        base_model.trainable = True

    # 4. Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    
    # EfficientNet sometimes benefits from a Dropout layer before the final dense, 
    # but to maintain STRICT comparison with ResNet101 baseline, we keep it identical: GAP -> Dense.
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"EfficientNet{subtype}_AD_Classifier")


"""
This file is used to build the model for RQ2 only
"""

import tensorflow as tf

def build_resnet101(input_shape=(224, 224, 3), num_classes=4, freeze_bn=True, weights='imagenet'):
    """
    Build a ResNet101 model for RQ2 that is consistent with RQ1, but parameterize freeze_bn for control.

    Args:
        input_shape (tuple): The shape of the input image.
        num_classes (int): The number of classes.
        weights (str): The weights to use for the model.
        freeze_bn (bool): Whether to freeze the batch normalization layers.
            This parameter determines how the model handles the standardization statistics of features during fine-tuning.
            If True, locks the BN layers' gamma (Scale) and beta (Shift) params, 
                and force using ImageNet pre-trained population statistics mu_{pop} and sigma^2_{pop}.
                This may cause Noisy Gradients on the direction of gradient descent.
            If False, allows BN layers to update gamma and beta via backpropagation, 
                and use the current batch statistics bar{x}_{batch} and s^2_{batch} for standardization, 
                while updating running statistics with momentum.

    Returns:
        tf.keras.Model: The built model.

    """
    inputs = tf.keras.Input(shape=input_shape)
    
    base_model = tf.keras.applications.ResNet101(
        include_top=False, 
        weights=weights, 
        input_tensor=inputs
    )
    
    # Implement RQ1 validated BN Freezing strategy
    if freeze_bn:
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet101_AD_Classifier")

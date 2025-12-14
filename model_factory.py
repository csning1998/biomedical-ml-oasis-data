import tensorflow as tf

def build_model(model_name='resnet101', num_classes=4, input_shape=(224, 224, 3)):
    """
    Factory function to build different backbones with CORRECT preprocessing baked in.
    
    Args:
        model_name (str): 'resnet101', 'resnet50', 'efficientnetv2b0', 'efficientnetb3'.
        num_classes (int): Number of output classes.
        input_shape (tuple): Input shape (H, W, C). Input is expected to be [0, 255] RGB.
        
    Returns:
        tf.keras.Model: Compiled-ready Keras model.
    """

    model_name_lower = model_name.lower()

    # 1. Define Input Layer (Raw RGB [0, 255])
    inputs = tf.keras.Input(shape=input_shape)

    # 2. Backbone
    if model_name_lower == 'resnet101':
        base_model = tf.keras.applications.ResNet101(
            include_top=False, 
            weights='imagenet', 
            input_tensor=inputs
        )
    elif model_name_lower == 'resnet50':
        base_model = tf.keras.applications.ResNet50(
            include_top=False, 
            weights='imagenet', 
            input_tensor=inputs
        )
    else:
        raise ValueError(f"For reproduction, only ResNet is supported currently. Got: {model_name}")

    # Classification Head: Freeze Backbone for Standard Transfer Learning
    base_model.trainable = False 
    
    # 3. Feature Aggregation such that (Batch, 7, 7, 2048) -> (Batch, 2048)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    # x = tf.keras.layers.Dropout(0.2)(x) # Temporarily disabled.

    # 4. Classification Head such that (Batch, 2048) -> (Batch, 4)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = tf.keras.Model(inputs, outputs, name=f"{model_name}_frozen")
    return model

# def build_flawed_model(num_classes=4):
#     """
#     A simplified builder specifically for the Flawed Baseline.
#     FEATURES:
#         - No Dropout (Maximal Memorization)
#         - Standard ResNet101 Backbone
#         - 'imagenet' weights
#     """
#     inputs = tf.keras.Input(shape=(224, 224, 3))
    
#     # 1. ResNet needs BGR + Zero-centering
#     x = tf.keras.layers.Lambda(
#         tf.keras.applications.resnet.preprocess_input,
#         name='resnet_preprocessing_flawed'
#     )(inputs)

#     # 2. Load Backbone
#     base_model = tf.keras.applications.ResNet101(
#         include_top=False, 
#         weights='imagenet', 
#         input_tensor=x
#     )
#     base_model.trainable = False 

#     # 3. Classification Head (No Dropout)
#     x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
#     outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

#     flawed_model = tf.keras.Model(inputs, outputs, name="ResNet101_Flawed_NoReg")
    
#     return flawed_model

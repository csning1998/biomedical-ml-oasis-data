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
    
    # 2. Dynamic Preprocessing Injection
    if 'resnet' in model_name_lower:
        """
        ResNet (Caffe Style): RGB -> BGR, Zero-centering
        A Lambda layer is used to ensure this becomes part of the model graph
        s.t. the model can be saved/loaded without external preprocessing
        """
        x = tf.keras.layers.Lambda(
            tf.keras.applications.resnet.preprocess_input,
            name='resnet_preprocessing'
        )(inputs)
        
        if model_name_lower == 'resnet101':
            base_model = tf.keras.applications.ResNet101(
                include_top=False, weights='imagenet', input_tensor=x)
        elif model_name_lower == 'resnet50':
            base_model = tf.keras.applications.ResNet50(
                include_top=False, weights='imagenet', input_tensor=x)
                
    elif 'efficientnet' in model_name_lower:
        """
        Given that EfficientNet V2 (TF Style): Expects [0, 255] or [0, 1].
        The V2 models typically include specific Rescaling/Normalization layers internally when include_top=False and weights='imagenet'.
        However, to be explicit and safe across TF versions, Assume that the V2 B0/B1... handles [0, 255] input if using 'imagenet' weights.
        If using V1 (e.g., EfficientNetB0), it might need Rescaling(1./255).
        """
        if 'v2' in model_name_lower:  # EfficientNetV2 expects [0, 255] input directly.
            x = inputs 
            if 'b0' in model_name_lower:
                base_model = tf.keras.applications.EfficientNetV2B0(
                    include_top=False, weights='imagenet', input_tensor=x)
            elif 'b1' in model_name_lower:
                base_model = tf.keras.applications.EfficientNetV2B1(
                    include_top=False, weights='imagenet', input_tensor=x)
        else: # EfficientNet V1 (B0-B7)
            x = tf.keras.layers.Rescaling(1./255)(inputs)
            if 'b0' in model_name_lower:
                base_model = tf.keras.applications.EfficientNetB0(
                    include_top=False, weights='imagenet', input_tensor=x)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Classification Head: Freeze Backbone for Standard Transfer Learning
    base_model.trainable = False 
    
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs, outputs, name=f"{model_name}_frozen")
    return model

def build_flawed_model(num_classes=4):
    """
    A simplified builder specifically for the Flawed Baseline.
    FEATURES:
        - No Dropout (Maximal Memorization)
        - Standard ResNet101 Backbone
        - 'imagenet' weights
    """
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # 1. ResNet needs BGR + Zero-centering
    x = tf.keras.layers.Lambda(
        tf.keras.applications.resnet.preprocess_input,
        name='resnet_preprocessing_flawed'
    )(inputs)

    # 2. Load Backbone
    base_model = tf.keras.applications.ResNet101(
        include_top=False, 
        weights='imagenet', 
        input_tensor=x
    )
    base_model.trainable = False 

    # 3. Classification Head (No Dropout)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    flawed_model = tf.keras.Model(inputs, outputs, name="ResNet101_Flawed_NoReg")
    
    return flawed_model

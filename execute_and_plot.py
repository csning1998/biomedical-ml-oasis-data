"""
This file is used to run the experiments for RQ2.
It contains the unified experiment runner and the plotting utils.
Refactored to support 'Restart Kernel' workflow by removing global variable dependencies.
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.ticker import MaxNLocator
from model_factory import build_resnet101
from data_loader import CLASS_MAP

def run_experiment(name, train_ds, val_ds, class_weights, epochs=20, save_dir='models', model_builder=None):
    """
    Unified experiment runner with Model Injection Support.

    Args:
        name (str): Experiment name (e.g., 'jet', 'viridis'). Used for file naming.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        class_weights (dict): Class weights for imbalanced training.
        epochs (int): Number of training epochs.c
        save_dir (str): Directory to save models and logs. Defaults to 'models'.
        model_builder (callable): Function to build the model (e.g., build_resnet101). 
            If None, defaults to build_resnet101 for backward compatibility.
    
    Returns:
        model: The trained Keras model.
        history: The history object or DataFrame.
    """
    
    # 1. Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"[{name}] Created directory: {save_dir}")

    # Define paths based on the 'name' argument to ensure isolation
    model_filename = f"resnet101_{name}_best.keras"
    model_path = os.path.join(save_dir, model_filename)
    csv_filename = f"history_{name}.csv"
    csv_path = os.path.join(save_dir, csv_filename)

    # 2. Check if model already exists (Resume or Skip)
    if os.path.exists(model_path):
        print(f"[{name}] Found existing model at {model_path}. Loading...")
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"[{name}] Model loaded successfully.")
            
            # Try to load history if it exists, for plotting
            if os.path.exists(csv_path):
                history = pd.read_csv(csv_path)
                print(f"[{name}] History loaded from {csv_path}.")
            else:
                history = None
                print(f"[{name}] Warning: Model found but no history CSV found at {csv_path}.")
                
            return model, history
        except Exception as e:
            print(f"[{name}] Error loading model: {e}. Will retrain.")
            # If load fails, fall through to training
    
    print(f"[{name}] Starting new training session...")
    
    # 3. Construct model (Dependency Injection)
    if model_builder is None:
        model_builder = build_resnet101 # Fallback for old notebooks
        
    # Note: Assume all builders accept 'freeze_bn' as defined in model_factory
    model = model_builder(freeze_bn=False)

    # 4. Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    
    # 5. Callbacks
    callbacks = [
        # 5a. Early Stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 5b. Model Checkpoint (Uses dynamic model_path)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # 5c. Reduce Learning Rate
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # 5d. CSV Logger (Uses dynamic csv_path)
        tf.keras.callbacks.CSVLogger(csv_path) 
    ]
    
    # 6. Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history

def plot_history(history, experiment_name):
    """
    Plot training curves.
    Args:
        history: model.fit() returns history object, or pd.DataFrame from CSVLogger.
        experiment_name (str): Used for title display.
    """
    # Compatibility handling: if Keras history object
    if hasattr(history, 'history'):
        metrics = history.history
    # Compatibility handling: if pd.DataFrame (from CSVLogger)
    elif isinstance(history, pd.DataFrame):
        metrics = history.to_dict(orient='list')
    elif isinstance(history, dict):
        metrics = history
    else:
        print("No history data available for plotting.")
        return

    acc_train = metrics.get('sparse_categorical_accuracy', [])
    acc_val = metrics.get('val_sparse_categorical_accuracy', [])
    
    if not acc_train:
        print(f"Accuracy metrics not found in history. Keys: {metrics.keys()}")
        return

    epochs = range(1, len(acc_train) + 1)

    # Construct DataFrame
    data = pd.DataFrame({
        'Epoch': list(epochs) + list(epochs),
        'Accuracy': acc_train + acc_val,
        'Set': ['Training'] * len(epochs) + ['Validation'] * len(epochs)
    })

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data, x='Epoch', y='Accuracy', hue='Set', style='Set',
        markers=True, dashes=False,
        palette={'Training': '#2ecc71', 'Validation': '#e74c3c'},
        ax=ax
    )

    ax.set_title(f"Experiment: {experiment_name} (Subject-level Split)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

def evaluate_and_plot_cm(model, val_ds, title_prefix="Model"):
    """
    Unified evaluation function: generate predictions, plot confusion matrix, and print classification report.
    
    Args:
        model: Trained Keras model
        val_ds: Validation or test dataset (tf.data.Dataset), suggested batch_size <= 16 to avoid OOM
        title_prefix: Prefix for the plot title (e.g., 'RQ1 Baseline', 'RQ2 Jet')
    """

    # 1. Get Predictions
    print("Generating predictions for Validation Set...")
    y_pred_probs = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 2. Get True Labels
    y_true = []
    for images, labels in val_ds:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)

    # 3. Define Class Names
    sorted_map = sorted(CLASS_MAP.items(), key=lambda item: item[1])
    class_names = [k for k, v in sorted_map]

    # 4. Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    # 5. Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix {title_prefix}\nValidation Accuracy: {np.mean(y_true == y_pred):.4f}')
    plt.show()

    # 6. Print Report
    print("\nClassification Report:\n")
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        labels=[0, 1, 2, 3], 
        zero_division=0
    ))

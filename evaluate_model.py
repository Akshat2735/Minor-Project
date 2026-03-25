import sys
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

from train_model import (
    load_dataset,
    preprocess_data,
    build_efficientnet_v2_model,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_CLASSES,
    DATA_DIR,
    CHECKPOINT_PATH,
    LOG_CSV_PATH,
)


def load_test_dataset(data_dir, img_size, batch_size):
    """Load dataset with same split as train_model and return test_ds and class_names."""
    train_ds, val_ds, test_ds = load_dataset(data_dir, img_size, batch_size)
    class_names = train_ds.class_names
    return test_ds, class_names

if __name__ == "__main__":
    # Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs detected:", gpus)
        try:
            # Currently, memory growth needs to be set before GPUs are initialized
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU devices found. Using CPU.")

    # Load and display training/validation metrics from CSV log
    print("\n--- Training and Validation Metrics ---")
    if os.path.exists(LOG_CSV_PATH):
        log_df = pd.read_csv(LOG_CSV_PATH)
        last_epoch_data = log_df.iloc[-1]
        print(f"Final Training Loss: {last_epoch_data['loss']:.4f}")
        print(f"Final Training Accuracy: {last_epoch_data['accuracy']:.4f}")
        print(f"Final Validation Loss: {last_epoch_data['val_loss']:.4f}")
        print(f"Final Validation Accuracy: {last_epoch_data['val_accuracy']:.4f}")
    else:
        print(f"Warning: Training log not found at {LOG_CSV_PATH}. Cannot display full training/validation history.")

    print("\nLoading test dataset...")
    test_ds, class_names = load_test_dataset(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    test_ds = test_ds.map(preprocess_data).prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Building model architecture...")
    model = build_efficientnet_v2_model(NUM_CLASSES, IMG_SIZE)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model weights from {CHECKPOINT_PATH}")
        model.load_weights(CHECKPOINT_PATH)
    else:
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}. Cannot evaluate model.")
        sys.exit(1)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    print(f"Evaluating model on the entire test set...")

    # Collect predictions and labels for the entire test set
    test_labels = []
    test_predictions = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        test_predictions.extend(np.argmax(predictions, axis=1))
        test_labels.extend(labels.numpy())

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    # Calculate accuracy for the entire test set
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report (Entire Test Set):")
    print(classification_report(test_labels, test_predictions, target_names=class_names))

    # Plot confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Entire Test Set)')
    plt.savefig('confusion_matrix_entire_test.png')
    print("Confusion matrix for entire test set saved as confusion_matrix_entire_test.png")
    plt.show()

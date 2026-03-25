
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# Define constants
IMG_SIZE = (224, 224) # EfficientNetV2M default input size
BATCH_SIZE = 32
NUM_CLASSES = 8 # Based on the 8 subdirectories
DATA_DIR = "kvasir-dataset-v2/kvasir-dataset-v2/"
CHECKPOINT_PATH = "./model_checkpoints/efficientnet_v2m_best_model.weights.h5"
LOG_CSV_PATH = "./training_log.csv"

# Load the dataset
def load_dataset(data_dir, img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3, # 70% for training, 30% for validation + test
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    # Split val_test_ds into validation and test sets (50/50 split of the 30%)
    val_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_ds = val_test_ds.take(val_batches // 2)
    test_ds = val_test_ds.skip(val_batches // 2)

    return train_ds, val_ds, test_ds

# Preprocessing function
def preprocess_data(image, label):
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label

# Build the model
def build_efficientnet_v2_model(num_classes, img_size):
    base_model = tf.keras.applications.EfficientNetV2M(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = True # Unfreeze the base model for fine-tuning

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = preprocess_data(inputs, inputs)[0] # Apply preprocessing to input
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    return model

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

    train_ds, val_ds, test_ds = load_dataset(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    # Apply preprocessing to datasets
    train_ds = train_ds.map(preprocess_data).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_data).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_data).prefetch(buffer_size=tf.data.AUTOTUNE)

    model = build_efficientnet_v2_model(NUM_CLASSES, IMG_SIZE)

    # Load weights from checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}")
        model.load_weights(CHECKPOINT_PATH)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower learning rate for fine-tuning
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.summary()

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True, # Save only weights to make loading easier
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5, # Stop if validation loss doesn't improve for 5 epochs
        restore_best_weights=True,
        verbose=1
    )

    csv_logger_callback = CSVLogger(
        LOG_CSV_PATH,
        separator=",",
        append=True
    )

    callbacks = [checkpoint_callback, early_stopping_callback, csv_logger_callback]

    history = model.fit(
        train_ds,
        epochs=20, # Increased epochs for fine-tuning
        validation_data=val_ds,
        callbacks=callbacks
    )

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_labels = np.concatenate([y for _, y in test_ds], axis=0)
    test_predictions_one_hot = model.predict(test_ds)
    test_predictions = np.argmax(test_predictions_one_hot, axis=1)

    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    # Print training and validation loss and accuracy
    print("\nTraining and Validation Metrics:")
    print(f"Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show() # This will display the plots.

import os
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

from train_model import (
    load_dataset,
    preprocess_data,
    build_efficientnet_v2_model,
    IMG_SIZE,
    NUM_CLASSES,
    DATA_DIR,
    CHECKPOINT_PATH,
)


def load_test_dataset():
    """
    Recreate the same split as in train_model.py and return only the test dataset
    plus the list of class names.
    """
    # One pass without split just to get class names in the right order.
    full_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=False,
    )
    class_names = full_ds.class_names

    # Use the same split logic as train_model.py
    train_ds, val_ds, test_ds = load_dataset(DATA_DIR, IMG_SIZE, batch_size=32)

    return test_ds, class_names


def build_model():
    """Build the EfficientNetV2M model and load trained weights if available."""
    model = build_efficientnet_v2_model(NUM_CLASSES, IMG_SIZE)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model weights from {CHECKPOINT_PATH}")
        model.load_weights(CHECKPOINT_PATH)
    else:
        print(
            f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}. "
            "Grad-CAM will use ImageNet-pretrained weights only."
        )

    return model


def get_backbone_and_dense(model: tf.keras.Model):
    """
    Get the EfficientNet backbone and the dense head from the full model.
    Grad-CAM uses the backbone's 4D output; we run backbone + GAP + dense manually in GradientTape.
    """
    backbone = model.get_layer("efficientnetv2-m")
    dense_layer = model.get_layer("dense")
    return backbone, dense_layer


def make_gradcam_heatmap(img_array, model, backbone, dense_layer, pred_index=None):
    """
    Generate a Grad-CAM heatmap using a manual forward pass (avoids Keras submodel graph issues).

    img_array: preprocessed image of shape (1, H, W, 3)
    model: full classification model (used only for pred_index if None)
    backbone: EfficientNetV2-M layer
    dense_layer: final Dense layer
    pred_index: optional class index; if None, uses the model's top predicted class.
    """
    with tf.GradientTape() as tape:
        x = tf.cast(img_array, tf.float32)
        conv_outputs = backbone(x, training=False)
        x_pooled = tf.keras.layers.GlobalAveragePooling2D()(conv_outputs)
        predictions = dense_layer(x_pooled)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_out, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / max_val, tf.zeros_like(heatmap))
    return heatmap.numpy()


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.4, cmap="jet"):
    """
    Overlay a Grad-CAM heatmap on top of the original image.

    original_image: tensor or array, shape (H, W, 3), values 0–255.
    heatmap: 2D array with values 0–1.
    """
    if isinstance(original_image, tf.Tensor):
        original_image = original_image.numpy()

    h, w = original_image.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (h, w)
    ).numpy().squeeze()

    try:
        colormap = plt.get_cmap(cmap)
    except AttributeError:
        colormap = cm.get_cmap(cmap)
    heatmap_rgb = colormap(heatmap_resized)[..., :3]

    if original_image.max() > 1.0:
        original_image = original_image / 255.0

    overlay = (1 - alpha) * original_image + alpha * heatmap_rgb
    return np.clip(overlay, 0.0, 1.0)


def save_gradcam_figure(image, heatmap_overlay, true_label, pred_label, class_names, out_path):
    """Save original image and Grad-CAM overlay side by side."""
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.max() > 1.0:
        image = image / 255.0

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_overlay)
    plt.axis("off")
    title = f"Grad-CAM\nTrue: {class_names[true_label]} | Pred: {class_names[pred_label]}"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(num_examples=5, output_dir="grad_cam_outputs"):
    # GPU/CPU info (same style as your other scripts)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("GPUs detected:", gpus)
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU devices found. Using CPU.")

    if not os.path.exists(CHECKPOINT_PATH):
        print(
            f"Warning: checkpoint not found at {CHECKPOINT_PATH}. "
            "If you already trained the model, please check the path."
        )

    print("Loading test dataset...")
    test_ds, class_names = load_test_dataset()
    # Keep original pixel values; we'll handle preprocessing per image.
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Building model and loading weights...")
    model = build_model()
    backbone, dense_layer = get_backbone_and_dense(model)
    print("Using EfficientNetV2-M backbone output for Grad-CAM.")

    os.makedirs(output_dir, exist_ok=True)

    # Collect some images from test set
    all_images = []
    all_labels = []
    for images, labels in test_ds:
        for i in range(images.shape[0]):
            all_images.append(images[i])
            all_labels.append(int(labels[i].numpy()))
        if len(all_images) >= num_examples * 3:
            break

    if not all_images:
        print("No images found in the test dataset.")
        return

    indices = list(range(len(all_images)))
    random.shuffle(indices)
    indices = indices[:num_examples]

    print(f"Generating Grad-CAM for {len(indices)} test images...")

    for idx, i in enumerate(indices, start=1):
        img = all_images[i]
        label = all_labels[i]

        # Prepare preprocessed batch
        img_batch = tf.expand_dims(img, axis=0)
        img_preprocessed, _ = preprocess_data(img_batch, tf.constant([label]))

        # Prediction
        preds = model.predict(img_preprocessed, verbose=0)
        pred_label = int(np.argmax(preds[0]))

        # Heatmap
        heatmap = make_gradcam_heatmap(
            img_preprocessed, model, backbone, dense_layer, pred_index=pred_label
        )
        overlay = overlay_heatmap_on_image(img, heatmap)

        out_name = f"gradcam_{idx}_true-{class_names[label]}_pred-{class_names[pred_label]}.png"
        out_path = os.path.join(output_dir, out_name)
        save_gradcam_figure(img, overlay, label, pred_label, class_names, out_path)
        print(f"Saved Grad-CAM visualization to {out_path}")


if __name__ == "__main__":
    main()


import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import optuna
import xgboost as xgb

from train_model import (
    load_dataset,
    preprocess_data,
    build_efficientnet_v2_model,
    IMG_SIZE,
    NUM_CLASSES,
    DATA_DIR,
    CHECKPOINT_PATH,
)


def build_feature_extractor():
    """
    Build the EfficientNetV2M model and return a feature extractor that outputs
    the penultimate representation (before the final Dense softmax layer).
    """
    model = build_efficientnet_v2_model(NUM_CLASSES, IMG_SIZE)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading EfficientNet weights from {CHECKPOINT_PATH}")
        model.load_weights(CHECKPOINT_PATH)
    else:
        print(
            f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}. "
            "Using ImageNet-pretrained backbone only."
        )

    # The second-to-last layer is the Dropout layer; we take its input (GAP output).
    gap_layer = model.layers[-3]  # GlobalAveragePooling2D
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=gap_layer.output,
        name="efficientnet_feature_extractor",
    )
    return feature_extractor


def dataset_to_features_labels(ds, feature_extractor):
    """
    Convert a tf.data.Dataset of images/labels into NumPy feature vectors and labels
    using the given feature_extractor model.
    """
    ds = ds.map(preprocess_data).prefetch(buffer_size=tf.data.AUTOTUNE)

    all_features = []
    all_labels = []

    for batch_images, batch_labels in ds:
        feats = feature_extractor(batch_images, training=False).numpy()
        all_features.append(feats)
        all_labels.append(batch_labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def run_optuna_xgboost_optimization(n_trials=10, results_dir=None):
    # Place results next to this script by default
    if results_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, "optuna_xgboost_results")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading train/val/test splits...")
    train_ds, val_ds, test_ds = load_dataset(DATA_DIR, IMG_SIZE, batch_size=32)

    print("Building EfficientNet feature extractor...")
    feature_extractor = build_feature_extractor()

    print("Extracting features for train/validation/test...")
    X_train, y_train = dataset_to_features_labels(train_ds, feature_extractor)
    X_val, y_val = dataset_to_features_labels(val_ds, feature_extractor)
    X_test, y_test = dataset_to_features_labels(test_ds, feature_extractor)

    print(f"Train features: {X_train.shape}, Val features: {X_val.shape}, Test features: {X_test.shape}")

    # Encode labels to a compact 0..K-1 range for XGBoost.
    # This avoids issues when some classes are missing in a split.
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    num_xgb_classes = len(le.classes_)

    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space for XGBoost
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "objective": "multi:softprob",
            "num_class": num_xgb_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
        }

        clf = xgb.XGBClassifier(**params)
        # Note: we intentionally do NOT pass eval_set here, because the
        # Keras-based split can yield slightly different class subsets in
        # train vs. val, which makes XGBoost's sklearn wrapper complain
        # about inconsistent classes between datasets.
        clf.fit(X_train, y_train_enc, verbose=False)

        y_val_pred_enc = clf.predict(X_val)
        acc = accuracy_score(y_val_enc, y_val_pred_enc)
        return acc  # we want to maximize validation accuracy

    study = optuna.create_study(direction="maximize")
    print(f"Starting Optuna optimization for {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (val accuracy): {best_trial.value:.4f}")
    print("  Params:")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    # Train final model on train+val with best params and evaluate on test set
    best_params = {
        **best_trial.params,
        "objective": "multi:softprob",
        "num_class": num_xgb_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
    }
    clf_best = xgb.XGBClassifier(**best_params)

    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full_enc = np.concatenate([y_train_enc, y_val_enc], axis=0)

    print("\nTraining best XGBoost model on train+val and evaluating on test...")
    clf_best.fit(X_train_full, y_train_full_enc, verbose=False)

    y_test_pred_enc = clf_best.predict(X_test)
    test_acc = accuracy_score(y_test_enc, y_test_pred_enc)
    test_f1 = f1_score(y_test_enc, y_test_pred_enc, average="weighted")
    # For the printed report, map back to original label indices.
    y_test_pred = le.inverse_transform(y_test_pred_enc.astype(int))
    report = classification_report(y_test, y_test_pred)

    print(f"\nTest Accuracy (XGBoost on EfficientNet features): {test_acc:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    print("\nClassification report:\n", report)

    # Save summary to a text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(results_dir, f"optuna_xgboost_summary_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Optuna + XGBoost optimization summary\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Best validation accuracy: {best_trial.value:.4f}\n")
        f.write("Best hyperparameters:\n")
        for k, v in best_trial.params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTest metrics (evaluated on held-out test set):\n")
        f.write(f"  Test accuracy: {test_acc:.4f}\n")
        f.write(f"  Test F1 (weighted): {test_f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)

    # Optionally save the trained XGBoost model
    model_path = os.path.join(results_dir, f"xgboost_best_model_{timestamp}.json")
    clf_best.save_model(model_path)

    print(f"\nSaved Optuna/XGBoost summary to {summary_path}")
    print(f"Saved best XGBoost model to {model_path}")


if __name__ == "__main__":
    run_optuna_xgboost_optimization()


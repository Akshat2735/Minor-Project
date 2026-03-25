# GI Image Classification Project

This project implements a gastrointestinal (GI) image classification system using deep learning and machine learning techniques. It leverages EfficientNetV2M for feature extraction and fine-tuning, followed by XGBoost for classification with hyperparameter optimization using Optuna. The system includes evaluation metrics, confusion matrices, and Grad-CAM visualizations for model interpretability.

## Features

- **Deep Learning Model**: Fine-tuned EfficientNetV2M for image classification
- **Feature Extraction**: Uses pre-trained EfficientNet to extract features from images
- **Machine Learning Classifier**: XGBoost optimized with Optuna for hyperparameter tuning
- **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and confusion matrix
- **Visualization**: Grad-CAM heatmaps for model explainability
- **Dataset**: Kvasir Dataset v2 for GI endoscopy images

## Dataset

The project uses the [Kvasir Dataset v2](https://datasets.simula.no/kvasir-v2/), which contains endoscopic images categorized into 8 classes:
- Dyed-lifted-polyps
- Dyed-resection-margins
- Esophagitis
- Normal-cecum
- Normal-pylorus
- Normal-z-line
- Polyps
- Ulcerative-colitis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gi-image-classification.git
   cd gi-image-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the EfficientNet Model**:
   ```bash
   python train_model.py
   ```
   This will train and fine-tune the EfficientNetV2M model on the dataset.

2. **Optimize XGBoost with Optuna**:
   ```bash
   python optuna_xgboost.py
   ```
   This extracts features using the trained EfficientNet and optimizes XGBoost hyperparameters.

3. **Evaluate the Model**:
   ```bash
   python evaluate_model.py
   ```
   This evaluates the XGBoost model on the test set and generates metrics and confusion matrix.

4. **Generate Grad-CAM Visualizations**:
   ```bash
   python grad_cam.py
   ```
   This creates Grad-CAM heatmaps for model interpretability.

## Results

- **XGBoost Validation Accuracy**: 99.55%
- **XGBoost Test Accuracy**: 99.19%
- **XGBoost Test F1-Score (Weighted)**: 98.99%

Detailed classification reports and confusion matrices are generated during evaluation.

## Project Structure

```
GI/
├── evaluate_model.py          # Model evaluation script
├── grad_cam.py                # Grad-CAM visualization
├── optuna_xgboost.py          # XGBoost with Optuna optimization
├── requirements.txt           # Python dependencies
├── train_model.py             # EfficientNet training script
├── training_log.csv           # Training logs
├── model_checkpoints/         # Saved model weights
├── grad_cam_outputs/          # Grad-CAM visualization outputs
├── kvasir-dataset-v2/         # Dataset directory
├── kvasir-dataset-v2-features/# Extracted features
└── optuna_xgboost_results/    # Optuna optimization results
```

## Dependencies

- TensorFlow
- NumPy
- Scikit-learn
- Matplotlib
- Optuna
- XGBoost

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Kvasir Dataset v2 by Simula Research Laboratory
- EfficientNetV2 by Google
- XGBoost and Optuna libraries
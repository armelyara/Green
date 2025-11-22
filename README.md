# Green: Traditional Plant Recognition Model

A deep learning model for recognizing traditional medicinal plants, powering the DrGreen mobile application. Built with MobileNetV2 and optimized for accurate multi-class plant classification.

## Overview

Green is a machine learning model designed to identify four traditional medicinal plants commonly used in herbal medicine. The model uses transfer learning with MobileNetV2 as the base architecture, combined with Focal Loss to handle class imbalance and extensive data augmentation to improve generalization.

### Recognized Plant Classes

The model can identify the following medicinal plants:

- **Artemisia** (Artemisia annua) - Known for antimalarial properties
- **Carica** (Carica papaya) - Papaya, used for digestive health
- **Goyavier** (Psidium guajava) - Guava, traditional remedy for various ailments
- **Kinkeliba** (Combretum micranthum) - West African medicinal plant

## Model Architecture

### Core Components

- **Base Model**: MobileNetV2 (pre-trained on ImageNet, frozen)
- **Input Size**: 224x224x3 RGB images
- **Loss Function**: Focal Loss (γ=2.0, α=0.25) with label smoothing (0.15)
- **Optimizer**: Adam with Cosine Decay learning rate schedule
- **Regularization**:
  - Dropout (60% and 30%)
  - L2 regularization (0.02)
  - Batch Normalization

### Architecture Details

```
Input (224x224x3)
    ↓
MobileNetV2 (frozen, ImageNet weights)
    ↓
Global Average Pooling
    ↓
Dropout (0.6)
    ↓
Dense (64 units, ReLU, L2 reg)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense (4 units, Softmax, L2 reg)
    ↓
Output (4 classes)
```

**Model Parameters:**
- Total parameters: 2,340,484
- Trainable parameters: 82,372
- Non-trainable parameters: 2,258,112

## Dataset

### Statistics

- **Total Images**: 1,164
- **Train/Validation Split**: 80/20 (stratified)
- **Training Images**: 931
- **Validation Images**: 233

### Class Distribution

| Class      | Total Images | Train | Validation | Percentage |
|------------|--------------|-------|------------|------------|
| Artemisia  | 275          | 220   | 55         | 23.6%      |
| Carica     | 356          | 285   | 71         | 30.6%      |
| Goyavier   | 241          | 193   | 48         | 20.7%      |
| Kinkeliba  | 292          | 233   | 59         | 25.1%      |

### Data Augmentation

To improve model robustness, the following augmentation techniques are applied during training:

- Random horizontal and vertical flips
- Random rotation (±30°)
- Random zoom (±20%)
- Random brightness adjustment (±20%)
- Random contrast adjustment (±20%)
- Random translation (±15%)

## Performance

### Model Metrics

- **Validation Accuracy**: 69.10%
- **Top-2 Accuracy**: 88.41%
- **Training Approach**: Transfer learning with frozen base
- **No Class Collapse**: Predictions are well-distributed across all classes

### Key Features

- **Stratified Splitting**: Ensures all classes are properly represented in both training and validation sets
- **Class Weighting**: Addresses class imbalance during training
- **Focal Loss**: Focuses on hard-to-classify examples
- **Early Stopping**: Prevents overfitting with patience of 15 epochs
- **Best Model Checkpoint**: Automatically saves the best performing model

## Setup and Installation

### Requirements

```bash
# Python 3.8 or higher
pip install tensorflow>=2.19.0
pip install numpy
pip install matplotlib
pip install seaborn
pip install pandas
pip install scikit-learn
pip install pillow
pip install gdown
```

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/armelyara/Green.git
cd Green
```

2. **Open in Google Colab**

Click the "Open in Colab" badge in the notebook or visit:
```
https://colab.research.google.com/github/armelyara/Green/blob/main/drgreen-v2.ipynb
```

3. **Enable GPU (Recommended)**
   - In Colab: Runtime → Change runtime type → GPU
   - Training is much faster with GPU acceleration

4. **Run the notebook**
   - The dataset will be automatically downloaded from Google Drive
   - Training will begin automatically after dataset preparation

### Platform Support

The notebook automatically detects and adapts to different environments:

- **Google Colab**: Full support with automatic path configuration
- **Kaggle Notebooks**: Full support with Kaggle-specific paths
- **Local Environment**: Supported with manual dataset setup

### Troubleshooting

#### Protobuf Compatibility Error

If you encounter `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, the notebook includes an automatic fix that installs the compatible protobuf version (3.20.3) in the first cell. Simply run all cells in order.

**Error example:**
```
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

**Solution:** The notebook automatically handles this by installing `protobuf==3.20.3` before importing TensorFlow.

#### Dataset Download Issues

If automatic download fails:

1. **Manual upload option**: Upload the dataset ZIP file directly to the notebook environment
2. **Check file ID**: Ensure the Google Drive file ID is correct and the file is publicly accessible
3. **Network issues**: Retry the download or use a different network connection

#### GPU Not Detected

- **Colab**: Runtime → Change runtime type → Hardware accelerator → GPU
- **Kaggle**: Settings → Accelerator → GPU T4 x2
- Training will work on CPU but will be significantly slower (10-20x)

## Training the Model

### Configuration

Key hyperparameters are defined in the `CONFIG` dictionary:

```python
CONFIG = {
    'img_height': 224,
    'img_width': 224,
    'batch_size': 16,
    'epochs': 100,
    'initial_lr': 0.0005,
    'validation_split': 0.2,
    'dropout_rate': 0.6,
    'num_classes': 4,
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,
    'label_smoothing': 0.15,
}
```

### Training Process

The notebook follows this workflow:

1. **Data Loading**: Downloads and extracts the plant image dataset
2. **Stratified Split**: Creates balanced train/validation sets using sklearn
3. **Data Pipeline**: Sets up TensorFlow data pipeline with augmentation
4. **Model Building**: Constructs MobileNetV2-based architecture
5. **Training**: Trains with Focal Loss, class weights, and callbacks
6. **Evaluation**: Generates confusion matrix and classification report
7. **Model Saving**: Saves best model checkpoint

### Callbacks

- **Early Stopping**: Stops training if validation accuracy doesn't improve for 15 epochs
- **Model Checkpoint**: Saves the best model based on validation accuracy
- **CSV Logger**: Records training metrics to CSV file

## Model Outputs

The trained model produces:

1. **Model File**: `models/best_model_v7.keras` - Best performing model
2. **Training Log**: `models/training_log_v7.csv` - Epoch-by-epoch metrics
3. **Visualizations**:
   - Training/validation accuracy curves
   - Training/validation loss curves
   - Top-2 accuracy curves
   - Confusion matrix
   - Per-class performance metrics

## Usage

### Making Predictions

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('models/best_model_v7.keras')

# Prepare an image
img_path = 'path/to/plant/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
class_names = ['artemisia', 'carica', 'goyavier', 'kinkeliba']
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0]) * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")
```

### Integration with DrGreen Mobile App

This model is designed to be integrated into the DrGreen mobile application for real-time plant recognition. The lightweight MobileNetV2 architecture ensures efficient inference on mobile devices.

**Deployment Options:**
- TensorFlow Lite for on-device inference
- TensorFlow Serving for cloud-based API
- Export to ONNX for cross-platform compatibility

## Project Structure

```
Green/
├── drgreen-v2.ipynb          # Main training notebook
├── models/                    # Saved models and logs
│   ├── best_model_v7.keras   # Best model checkpoint
│   └── training_log_v7.csv   # Training metrics
├── README.md                  # This file
└── LICENSE                    # Project license
```

## Key Improvements in V2

The current version (V2) includes critical improvements over the initial version:

- **Stratified Split**: Uses sklearn's `train_test_split` with stratification to ensure all classes are represented in validation
- **Focal Loss**: Addresses class imbalance by focusing on hard examples
- **Balanced Predictions**: No class collapse - predictions are well-distributed across all 4 classes
- **Improved Regularization**: Multiple dropout layers and L2 regularization prevent overfitting
- **Class Weighting**: Dynamic class weights during training to handle dataset imbalance

## Future Improvements

Potential enhancements for future versions:

- [ ] Expand dataset with more plant species
- [ ] Implement data collection pipeline for continuous learning
- [ ] Add explainability features (Grad-CAM visualization)
- [ ] Fine-tune base model layers for improved accuracy
- [ ] Implement ensemble methods
- [ ] Add confidence thresholding for uncertain predictions
- [ ] Support for plant part recognition (leaf, flower, stem)
- [ ] Multi-label classification for mixed plant images

## Citation

If you use this model in your research or application, please cite:

```
Green: Traditional Plant Recognition Model
Repository: https://github.com/armelyara/Green
Model: MobileNetV2 + Focal Loss for Traditional Plant Classification
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Pre-trained MobileNetV2 weights from ImageNet
- TensorFlow and Keras teams for the excellent deep learning framework
- Google Colab for providing free GPU resources
- Contributors to the plant image dataset

## Contact

For questions, issues, or collaboration opportunities related to the Green model or DrGreen application, please open an issue on this repository.

---

**Note**: This model is intended for educational and research purposes. For medical or health-related decisions, always consult qualified healthcare professionals.

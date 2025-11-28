# CNN Image Classification Project

## Project Overview

This project implements three CNN approaches for a custom image dataset. The goal is to classify **AirPods vs Magic Mouse** images using different neural network architectures and compare their performance.

**Student Information:**

- **Name:** Eren Ali Koca
- **Student ID:** 2212721021
- **Course:** BLG-407 Machine Learning
- **GitHub Repo:** https://github.com/erennali/CNN_airpods_magicmouse

---

## Dataset

This project classifies **AirPods vs Magic Mouse** using custom-collected images.

- **Class 1:** AirPods (67 images)
- **Class 2:** Magic Mouse (63 images)
- **Total:** 130 original images
- All images are original photos taken with phone camera
- Images automatically resized to 128x128 pixels (Model 2 & 3) / 224x224 (Model 1)

**Folder Structure:**

```
dataset/
├── airpods/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── magic_mouse/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

---

## Notebooks

| Notebook       | Description                          | Architecture                | Test Accuracy |
| -------------- | ------------------------------------ | --------------------------- | ------------- |
| `model1.ipynb` | Transfer Learning with VGG16         | VGG16 + Custom Dense Layers | **100%**      |
| `model2.ipynb` | Basic CNN trained from scratch       | 3 Conv + 2 Dense Layers     | **88%**       |
| `model3.ipynb` | Hyperparameter tuning & optimization | 4 Conv + BatchNorm + Dense  | **96%**       |

---

## Model Details

### Model 1: Transfer Learning (VGG16)

- Uses ImageNet pretrained weights
- Fine-tuning with frozen early layers
- Data augmentation applied
- **Test Accuracy: 100%**

### Model 2: Basic CNN (From Scratch)

- Simple 3-layer CNN architecture
- No pretrained weights
- Trained from scratch on custom dataset
- **Test Accuracy: 88%**

### Model 3: Improved CNN (Hyperparameter Tuning)

- Enhanced architecture with BatchNormalization
- Systematic hyperparameter experimentation (8 experiments)
- Optimized based on experiment results
- **Test Accuracy: 96%** (+8% improvement over Model 2)

---

## Model 3: Hyperparameter Tuning Results

In Model 3, we conducted **8 different hyperparameter experiments** to find the optimal configuration. The goal was to improve Model 2's performance through systematic optimization.

### Experiment Results Table

| Exp # | Experiment        | Batch Size | Filters          | Dropout | Learning Rate | Augmentation | Extra Layer | Test Accuracy |
| ----- | ----------------- | ---------- | ---------------- | ------- | ------------- | ------------ | ----------- | ------------- |
| 1     | Baseline          | 32         | 32-64-128        | 0.5     | 0.001         | No           | No          | 84%           |
| 2     | Increased Filters | 32         | 64-128-256       | 0.5     | 0.001         | No           | No          | 76%           |
| 3     | Batch Size 64     | 64         | 32-64-128        | 0.5     | 0.001         | No           | No          | **92%**       |
| 4     | LR 0.0005         | 32         | 32-64-128        | 0.5     | 0.0005        | No           | No          | 52%           |
| 5     | Dropout 0.3       | 32         | 32-64-128        | 0.3     | 0.001         | No           | No          | 76%           |
| 6     | Extra Conv Layer  | 32         | 32-64-128-256    | 0.5     | 0.001         | No           | Yes         | 84%           |
| 7     | With Augmentation | 32         | 32-64-128        | 0.5     | 0.001         | Yes          | No          | 80%           |
| 8     | Combined Best     | 64         | 64-128-256+Extra | 0.5     | 0.001         | No           | Yes         | 84%           |

**Final Best Model (Extended Training):** **96% Test Accuracy**

### Key Findings

1. **Batch Size Impact:** Increasing batch size from 32 to 64 provided the best single-parameter improvement (92%)
2. **Learning Rate Sensitivity:** Lower learning rate (0.0005) was detrimental for small datasets (52%)
3. **Data Augmentation:** Did not provide expected benefits on this small dataset
4. **Extra Convolutional Layer:** Added model depth but required careful tuning
5. **Combined Parameters:** Best results achieved with batch_size=64, larger filters, and extra conv layer

### Model 3 Final Architecture

```
Input (128x128x3)
    ↓
Conv2D (64 filters, 3x3) → BatchNormalization → MaxPooling2D
    ↓
Conv2D (128 filters, 3x3) → BatchNormalization → MaxPooling2D
    ↓
Conv2D (256 filters, 3x3) → BatchNormalization → MaxPooling2D
    ↓
Conv2D (256 filters, 3x3) → BatchNormalization → MaxPooling2D [Extra Layer]
    ↓
Flatten → Dense(256, ReLU) → Dropout(0.5)
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(2, Softmax) → Output
```

### Training Configuration

- **Optimizer:** Adam (learning_rate=0.001)
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 64
- **Epochs:** 50 (with EarlyStopping)
- **Callbacks:** EarlyStopping (patience=7), ReduceLROnPlateau (patience=4)

---

## Model Comparison Summary

| Metric             | Model 1 (VGG16)   | Model 2 (Basic) | Model 3 (Improved) |
| ------------------ | ----------------- | --------------- | ------------------ |
| Architecture       | Transfer Learning | From Scratch    | Optimized CNN      |
| Conv Layers        | 16 (VGG16)        | 3               | 4                  |
| Pretrained         | Yes (ImageNet)    | No              | No                 |
| BatchNormalization | No                | No              | Yes                |
| Test Accuracy      | **100%**          | 88%             | **96%**            |
| Improvement        | -                 | Baseline        | +8% over Model 2   |

**Conclusion:** Model 3 achieved **96% accuracy**, demonstrating an **8% improvement over Model 2 (88%)** through systematic hyperparameter optimization. While Model 1 (VGG16 Transfer Learning) achieved 100%, Model 3 shows that a custom CNN can achieve competitive results with proper tuning.

---

## Installation

Use the provided `requirements.txt` to install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Training (Google Colab Recommended)

1. Upload notebooks to Google Colab
2. Mount Google Drive and set project path
3. Ensure dataset is in `dataset/airpods/` and `dataset/magic_mouse/`
4. Run cells sequentially
5. Models will be saved as `.h5` files

### Testing Models

After training all three models, use the camera application to test them:

```bash
python camera_app.py
```

### Camera App Features

- **🔍 Single Model Test**: Test one model at a time with camera or uploaded images
- **⚖️ Compare Mode**: Test all three models simultaneously and compare results
- **📦 Object Detection**: Automatic bounding box detection around objects
  - Smart Detection (auto-selects best method)
  - Contour Detection (edge-based)
  - Center Focus (assumes centered object)
- **🎯 Visual Bounding Box**: Professional rectangle highlighting with corner markers
- **⚡ Real-time Performance**: Shows FPS, inference time (ms) for each model
- **🎨 Color-coded Results**: Each model has unique color for easy identification
- **Model Selection**: Switch between Model 1 (VGG16), Model 2 (Basic CNN), and Model 3 (Optimized CNN)

### Test Options

#### 1. Camera Test (Real-time)

1. Click "▶ Start Camera" button
2. Point camera at AirPods or Magic Mouse
3. Watch real-time predictions with bounding box
4. FPS counter shows performance
5. Click "⏸ Stop Camera" when done

#### 2. Image Upload Test

1. Click "📁 Upload Image" button
2. Select an image file (JPG, PNG, BMP)
3. Object is automatically detected with bounding box
4. Prediction results shown instantly

#### 3. Detection Settings

- **Show Bounding Box**: Toggle rectangle highlighting on/off
- **Detection Method**:
  - Smart (Auto): Best results, auto-selects method
  - Contour Detection: Edge-based, works well with clear backgrounds
  - Center Focus: Assumes object is centered

#### 4. Test Modes

- **Single Model Test**: Focus on one model, detailed metrics
- **Compare All Models**: See all three models' predictions side-by-side

### Command-Line Testing

For quick model comparison without GUI:

```bash
python test_models.py
```

This interactive tool provides:

- Single image testing with all models
- Batch testing from dataset folders
- Performance comparison (accuracy and inference time)
- Detailed metrics for each model

---

## Project Structure

```
CNN_airpods_magicmouse/
├── dataset/
│   ├── airpods/           # AirPods images (67)
│   └── magic_mouse/       # Magic Mouse images (63)
├── model1.ipynb           # VGG16 Transfer Learning
├── model2.ipynb           # Basic CNN from scratch
├── model3.ipynb           # Hyperparameter tuning experiments
├── model1_transfer_learning.h5    # Saved Model 1
├── model2_basic_cnn.h5            # Saved Model 2
├── model3_improved_cnn.h5         # Saved Model 3
├── camera_app.py          # Real-time testing application
├── test_models.py         # CLI testing tool
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Key Technical Features

### Advanced Object Detection

- **Multi-method Detection**: Implements 3 different object detection algorithms
- **Contour-based Detection**: Uses OpenCV edge detection and morphological operations
- **Smart Auto-selection**: Automatically chooses best detection method
- **Professional Bounding Box**: Rectangle with corner markers for clean visualization

### Performance Optimization

- **Threaded Camera Processing**: Non-blocking camera capture with queue management
- **Real-time FPS Tracking**: Monitor application performance
- **Memory Leak Prevention**: Proper resource cleanup and thread management
- **Efficient Preprocessing**: Optimized image preprocessing pipeline

### Robust Error Handling

- **Graceful Degradation**: App continues running even if some models fail to load
- **Thread Safety**: Thread locks prevent race conditions
- **Camera Error Recovery**: Handles camera disconnection and errors
- **Comprehensive Logging**: Debug information for troubleshooting

### User Experience

- **Intuitive GUI**: Modern dark theme with clear visual hierarchy
- **Color-coded Models**: Each model has unique color for easy identification
- **Real-time Feedback**: Live predictions with confidence scores
- **Flexible Testing**: Camera or image upload, single or comparison mode

---

## Commits and Style

- Commit messages are short, English and prefixed (e.g. `feat:`, `chore:`, `docs:`)
- Code follows PEP 8 style guidelines
- Notebooks contain detailed explanations in markdown cells

---

## Notes

- All notebooks were run on Google Colab with GPU (T4)
- Hyperparameter tuning table screenshot is included in model3.ipynb (as required)
- Models are saved in HDF5 format (.h5)
- The camera app requires a webcam for real-time testing

---

## Author

**Eren Ali Koca**

- Student ID: 2212721021
- Course: BLG-407 Machine Learning
- Project: CNN Image Classification (AirPods vs Magic Mouse)
- GitHub: https://github.com/erennali/CNN_airpods_magicmouse

---

## Project Statistics

| Metric                            | Value                    |
| --------------------------------- | ------------------------ |
| Total Training Images             | 130                      |
| Number of Classes                 | 2 (AirPods, Magic Mouse) |
| CNN Models Implemented            | 3                        |
| Hyperparameter Experiments        | 8                        |
| Best Accuracy (Transfer Learning) | 100% (Model 1)           |
| Best Accuracy (Custom CNN)        | 96% (Model 3)            |
| Improvement over Baseline         | +8% (Model 3 vs Model 2) |

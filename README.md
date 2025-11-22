# CNN Image Classification Project

## Project Overview

This project implements three CNN approaches for a custom image dataset. Notebooks are prepared so you can run training after adding real images to `dataset/`.

## Dataset

This project classifies **AirPods vs Magic Mouse** using custom-collected images.

- **Class 1:** AirPods (50+ images)
- **Class 2:** Magic Mouse (50+ images)
- All images are original photos taken with phone camera
- Images automatically resized to 128x128 pixels
- Folder structure:

```
dataset/
	airpods/
		img1.jpg
		img2.jpg
		...
	magic_mouse/
		img1.jpg
		img2.jpg
		...
```

## Notebooks

- `model1.ipynb`: Transfer learning with VGG16 (uses ImageNet weights)
- `model2.ipynb`: Basic CNN trained from scratch
- `model3.ipynb`: Hyperparameter tuning, data augmentation and final model

## Installation

Use the provided `requirements.txt` to install dependencies:

```bash
pip install -r requirements.txt
```

## How to run

1. Put your images into `dataset/airpods/` and `dataset/magic_mouse/`.
2. Open `model1.ipynb` and fill student info at the top cells.
3. Run cells sequentially. If dataset is small, reduce batch size and epochs.
4. After experiments, open `model3.ipynb` to view hyperparameter table and train best model.

## Testing Models

After training all three models, use the camera application to test them:

```bash
python camera_app.py
```

### Camera App Features:
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

### Test Options:

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

## Commits and style

- Commit messages are short, English and prefixed (e.g. `feat:`, `chore:`, `docs:`).

## Notes

- Notebooks contain placeholders for student info and GitHub link. Replace them before submission.
- Add your GitHub repo URL to the top of each notebook.

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

## Author

**Eren Ali Koca**

- Student ID: 2212721021
- Course: BLG-407 Machine Learning
- Project: CNN Image Classification (AirPods vs Magic Mouse)
- GitHub: https://github.com/erenalikoca/CNN_airpods_magicmouse

## Project Statistics

- **3 CNN Models**: Transfer Learning, Basic CNN, Optimized CNN
- **130 Training Images**: 67 AirPods + 63 Magic Mouse
- **8 Hyperparameter Experiments**: Systematic optimization in Model 3
- **Best Accuracy**: 100% (Model 1 - VGG16 Transfer Learning)
- **Advanced Features**: Real-time object detection with bounding boxes

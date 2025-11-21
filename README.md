# CNN Image Classification Project

## Project Overview
This project implements three CNN approaches for a custom image dataset. Notebooks are prepared so you can run training after adding real images to `dataset/`.

## Dataset requirements
- Minimum 2 classes
- At least 50 images per class (100+ total)
- Images must be original photos taken by you
- Image size: recommended 128x128 (min 64x64)
- Folder structure:

```
dataset/
	class1/
		img1.jpg
	class2/
		img1.jpg
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
1. Put your images into `dataset/class1/` and `dataset/class2/`.
2. Open `model1.ipynb` and fill student info at the top cells.
3. Run cells sequentially. If dataset is small, reduce batch size and epochs.
4. After experiments, open `model3.ipynb` to view hyperparameter table and train best model.

## Commits and style
- Commit messages are short, English and prefixed (e.g. `feat:`, `chore:`, `docs:`).

## Notes
- Notebooks contain placeholders for student info and GitHub link. Replace them before submission.
- Add your GitHub repo URL to the top of each notebook.

## Author
Student project for Machine Learning course (BLG407)

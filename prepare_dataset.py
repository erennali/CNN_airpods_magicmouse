import os
import shutil
from pathlib import Path

def organize_dataset(source_dir, target_dir, class_names):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    for class_name in class_names:
        class_dir = os.path.join(target_dir, class_name)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {class_dir}")
    
    print(f"\nDataset structure created successfully!")
    print(f"Please add your images to the respective class folders:")
    for class_name in class_names:
        print(f"  - {target_dir}/{class_name}/")

if __name__ == "__main__":
    class_names = ["class1", "class2"]
    organize_dataset(".", "dataset", class_names)

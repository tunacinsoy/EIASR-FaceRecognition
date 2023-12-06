import os
import shutil
from sklearn.model_selection import train_test_split

# Path to the dataset directory
dataset_dir = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\GeorgeBush"
print(dataset_dir)
# Collect all image filenames
image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

# Split the dataset
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Create directories for the training and testing sets
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Move files to the respective directories
for file in train_files:
    shutil.move(os.path.join(dataset_dir, file), train_dir)

for file in test_files:
    shutil.move(os.path.join(dataset_dir, file), test_dir)

print("Dataset divided into training and testing sets.")

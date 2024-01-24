import os
import shutil
import random

def split_data_without_subfolders(source_directory, train_directory, test_directory, test_ratio=0.2):
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    for individual_folder in os.listdir(source_directory):
        individual_path = os.path.join(source_directory, individual_folder)
        if os.path.isdir(individual_path):
            files = [f for f in os.listdir(individual_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(files)

            test_count = max(1, int(len(files) * test_ratio))  # Ensure at least one image for testing
            test_files = files[:test_count]
            train_files = files[test_count:]

            for file in test_files:
                shutil.copy(os.path.join(individual_path, file), os.path.join(test_directory, file))

            for file in train_files:
                shutil.copy(os.path.join(individual_path, file), os.path.join(train_directory, file))

source_directory = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\outputsOfEnhanced"
train_directory = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\trainingDataset"
test_directory = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\testingDataset"

split_data_without_subfolders(source_directory, train_directory, test_directory, test_ratio=0.2)

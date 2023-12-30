import os
import shutil

def create_subfolders(source_path, destination_path):
    # Ensure the destination directory exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # List all entries in the source directory
    for entry in os.listdir(source_path):
        full_path = os.path.join(source_path, entry)

        # Check if the entry is a directory
        if os.path.isdir(full_path):
            # Create a corresponding subfolder in the destination
            dest_subfolder = os.path.join(destination_path, entry)
            if not os.path.exists(dest_subfolder):
                os.makedirs(dest_subfolder)

# Example usage
source_path = 'C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\subjects'
destination_path = 'C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\outputsOfEnhanced'
create_subfolders(source_path, destination_path)

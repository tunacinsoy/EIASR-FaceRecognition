import os
import glob

def count_images_in_subfolders(root_folder):
    image_extensions = ['*.jpg']
    totalImageCount = 0
    for folder, subfolders, files in os.walk(root_folder):
        image_count = 0
        for extension in image_extensions:
            totalImageCount += len(glob.glob(os.path.join(folder, extension)))
            image_count += len(glob.glob(os.path.join(folder, extension)))
        #print(f"Folder '{folder}' contains {image_count} image(s)")
    return totalImageCount

subjects_folder = 'C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\subjects'
total_images_in_subjects = count_images_in_subfolders(subjects_folder)
print(f"Total number of images in subjects folder: {total_images_in_subjects}")




v1_outputs_folder = 'C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\outputs'
v1_total_images_in_outputs = count_images_in_subfolders(v1_outputs_folder)
print(f"Total number of images in V1.0 outputs folder: {v1_total_images_in_outputs}")

v1_suitable_outputs_folder = 'C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\suitableOutputs'
v1_total_images_in_suitable_outputs = count_images_in_subfolders(v1_suitable_outputs_folder)
print(f"Total number of suitable images V1.0 suitableOutputs folder: {v1_total_images_in_suitable_outputs}")

print(f" V1: Ratio of Obtained Images over All Images of Subjects: %{(v1_total_images_in_outputs / total_images_in_subjects) * 100}")
print(f" V1: Ratio of Suitable Images: %{(v1_total_images_in_suitable_outputs / v1_total_images_in_outputs) * 100}")

v2_outputs_folder = 'C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\outputsOfEnhanced'
v2_total_images_in_outputs = count_images_in_subfolders(v2_outputs_folder)
print(f"Total number of images in V2.0 outputs folder: {v2_total_images_in_outputs}")

v2_suitable_outputs_folder = 'C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\suitableOutputsOfEnhanced'
v2_total_images_in_suitable_outputs = count_images_in_subfolders(v2_suitable_outputs_folder)
print(f"Total number of suitable images V2.0 suitableOutputs folder: {v2_total_images_in_suitable_outputs}")

print(f" V2: Ratio of Obtained Images over All Images of Subjects: %{(v2_total_images_in_outputs / total_images_in_subjects) * 100}")
print(f" V2: Ratio of Suitable Images: %{(v2_total_images_in_suitable_outputs / v2_total_images_in_outputs) * 100}")
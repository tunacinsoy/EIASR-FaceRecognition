import cv2
import os
import numpy as np
import shutil

def is_image_64x64_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.shape == (64, 64)

def contains_only_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) == 1

def copy_to_folder(source, destination):
    shutil.copy(source, destination)

def process_images(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_directory, filename)
            if is_image_64x64_grayscale(image_path) and contains_only_face(image_path):
                print(f"Image {filename} is suitable.")
                copy_to_folder(image_path, os.path.join(output_directory, filename))
            else:
                print(f"Image {filename} is not suitable.")

# Replace these paths with your input and output directories
input_directory = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\outputs"
output_directory = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\suitableOutputs"

process_images(input_directory, output_directory)

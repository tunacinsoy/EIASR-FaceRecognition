import cv2
import os
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

def process_images(input_directory, output_directory):
    for person_name in os.listdir(input_directory):
        person_path = os.path.join(input_directory, person_name)
        if os.path.isdir(person_path):
            output_person_path = os.path.join(output_directory, person_name)
            if not os.path.exists(output_person_path):
                os.makedirs(output_person_path)

            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_path, filename)
                    if is_image_64x64_grayscale(image_path) and contains_only_face(image_path):
                        print(f"Image {filename} of {person_name} is suitable.")
                        shutil.copy(image_path, output_person_path)
                    else:
                        print(f"Image {filename} of {person_name} is not suitable.")

# Replace these paths with your input and output directories
input_directory = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\outputsOfEnhanced"
output_directory = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\suitableOutputsOfEnhanced"

process_images(input_directory, output_directory)

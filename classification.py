import os
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from FaceClassifier import FaceClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Takes the path to a folder that contains face images as an input,
# and returns a Numpy array that contains the data for the faces, and a list with the corresponding labels.
def get_dataset(image_folder_path):
    image_data = []
    labels = []

    for filename in os.listdir(image_folder_path):
        labels.append('_'.join(filename.split("_")[:-2]))
        image = Image.open(os.path.join(image_folder_path, filename))
        image_as_array = np.array(image).flatten()
        image_data.append(image_as_array)

    return (np.array(image_data), labels)

def get_accuracy(actual_labels, predicted_labels):
    num_classified = len(predicted_labels)
    num_correctly_classified = 0
    for i in range(num_classified):
        if predicted_labels[i] == actual_labels[i]:
            num_correctly_classified += 1
    accuracy = num_correctly_classified / num_classified
    
    return accuracy

def print_classification_info(actual_labels, predicted_labels):
    num_classified = len(predicted_labels)
    for i in range(num_classified):
        if predicted_labels[i] == actual_labels[i]:
            print(f"Successfully classified {actual_labels[i]}.")
        else:
            print(f"Misclassified {actual_labels[i]} as {predicted_labels[i]}.")

def plot_confusion_matrix(actual_labels, predicted_labels):
    cm = confusion_matrix(actual_labels, predicted_labels)
    sns.heatmap(cm, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.show()

def main():
    # olivetti_faces = fetch_olivetti_faces()
    # training_data, test_data, training_labels, test_labels = train_test_split(olivetti_faces['data'], olivetti_faces['target'], test_size = 0.2, random_state = 17)
    
    training_data, training_labels = get_dataset('trainingDataset')
    test_data, test_labels = get_dataset('testingDataset')

    face_classifier = FaceClassifier()
    face_classifier.train(training_data, training_labels, 0.95)
    predicted_labels = face_classifier.classify(test_data)

    print(get_accuracy(test_labels, predicted_labels))
    # print_classification_info(test_labels, predicted_labels)
    # plot_confusion_matrix(test_labels, predicted_labels)

if __name__ == "__main__":
    main()
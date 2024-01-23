import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import uuid

class FaceClassifier:
    def __init__(self):
        self.average_face_training = None
        self.eigenfaces = None
        self.stored_face_weights = None
        self.stored_face_labels = None

    # Displays a face vector as an image.
    def visualize_face(face_vector):
        n_pixels = int(math.sqrt(face_vector.shape[0]))
        plt.imshow(face_vector.reshape(n_pixels, n_pixels), cmap='gray')
        plt.show()

    def __get_average_face_vector(self, face_matrix):
        num_pixels_per_image = face_matrix.shape[1]
        num_images = face_matrix.shape[0]

        face_sum_vector = np.zeros(num_pixels_per_image)
        for face_vector in face_matrix:
            face_sum_vector += face_vector

        return np.divide(face_sum_vector, num_images)
    
    # Assumes that face_matrix is a 2D array where each row represents a face.
    # The threshold specifies what value the cumulative_explained_variance should reach when selecting the number of principal components.
    def __get_eigenfaces(self, face_matrix, threshold):
        # Centers the data by subtracting the average_face_vector from each row.
        centered_face_matrix = face_matrix - self.average_face_training

        # Finds eigenvectors and eigenvalues by using Singular Value Decomposition.
        eigenvectors, singular_values, U_t = np.linalg.svd(np.transpose(centered_face_matrix))
        # Only a subset of the eigenvalues of the covariance matrix can be gotten by using Singular Value Decomposition. However, this subset contains all meaningful eigenvalues. 
        # According to M. Turk and A. Pentland, there are (num_images - 1) non-zero eigenvalues when the number of images is less than the number of pixels per image.
        eigenvalues = np.square(singular_values)

        # Dimensionality reduction is achieved by selecting a subset of the principal components.
        # The selected number of principal components is determined based on a threshold for the cumulative explained variance.
        total_sum_of_eigenvalues = sum(eigenvalues)
        sum_of_eigenvalues = 0
        num_principal_components = 0
        
        for eigenvalue in eigenvalues:
            num_principal_components += 1
            sum_of_eigenvalues += eigenvalue
            cumulative_explained_variance = sum_of_eigenvalues / total_sum_of_eigenvalues
            if (cumulative_explained_variance >= threshold):
                break

        # Each column is a principal components, in this case, an eigenface. Keep as many columns as num_principal_components.
        return eigenvectors[:, :num_principal_components]
    
    # Calculates the weights of each face, in other words the multidimensional points that represent the faces.
    def __calculate_weights(self, face_matrix):
        weights = []
        for index in range(len(face_matrix)):
            face = face_matrix[index, :]
            weights.append(self.__calculate_single_set_of_weights(face))

        return weights
    
    def __calculate_single_set_of_weights(self, face):
        face_centered = face - self.average_face_training
        current_weights = np.matmul(face_centered, self.eigenfaces)

        return current_weights

    def train(self, face_matrix, labels, threshold):
        self.average_face_training = self.__get_average_face_vector(face_matrix)
        self.eigenfaces = self.__get_eigenfaces(face_matrix, threshold)
        self.stored_face_weights = self.__calculate_weights(face_matrix)
        self.stored_face_labels = labels.copy()

    # Each face that is provided as input is given the label of the training face with the most similar set of weights.
    def classify(self, face_matrix):
        predicted_labels = []
        for index in range(len(face_matrix)):
            current_set_of_weights = self.__calculate_single_set_of_weights(face_matrix[index, :])
            distances = cdist(current_set_of_weights.reshape(1, -1), self.stored_face_weights)
            nearest_neighbor_index = np.argmin(distances)
            predicted_label = self.stored_face_labels[nearest_neighbor_index]
            predicted_labels.append(predicted_label)

        return predicted_labels
    
    # For each face provided as input, it is checked if the distance between the computed weights of the input face, 
    # and the nearest set of weights are within a specified distance. If it is, the input face is given the label of the face with the most similar weights.
    # Otherwise, the input face is considered to belong to a new subject and is labeled with a unique id. 
    # The calculated set of weights, and the new label are then stored, together with the weights and labels of the training faces, and are used when classifying subsequent faces.
    def classify_dynamically(self, face_matrix, max_nearest_neighbor_distance):
        predicted_labels = []
        for index in range(len(face_matrix)):
            current_set_of_weights = self.__calculate_single_set_of_weights(face_matrix[index, :])
            distances = cdist(current_set_of_weights.reshape(1, -1), self.stored_face_weights)
            nearest_neighbor_distance = np.min(distances)
            
            if (nearest_neighbor_distance <= max_nearest_neighbor_distance):
                nearest_neighbor_index = np.argmin(distances)
                predicted_label = self.stored_face_labels[nearest_neighbor_index]
            else:
                new_label = uuid.uuid4()
                self.stored_face_weights.append(current_set_of_weights)
                self.stored_face_labels.append(new_label)
                predicted_label = new_label
            predicted_labels.append(predicted_label)

        return predicted_labels
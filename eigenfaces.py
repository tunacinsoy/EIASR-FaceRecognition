from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt
import math

# Displays a face vector as an image.
def visualize_face(face_vector):
    n_pixels = int(math.sqrt(face_vector.shape[0]))
    plt.imshow(face_vector.reshape(n_pixels, n_pixels), cmap='gray')
    plt.show()

def get_average_face_vector(face_matrix):
    num_pixels_per_image = face_matrix.shape[1]
    num_images = face_matrix.shape[0]

    face_sum_vector = np.zeros(num_pixels_per_image)
    for face_vector in face_matrix:
        face_sum_vector += face_vector
    return np.divide(face_sum_vector, num_images)

# Assumes that face_matrix is a 2D array where each row represents a face.
# The threshold specifies what value the cumulative_explained_variance should reach when selecting the number of principal components.
def get_eigenfaces(face_matrix, threshold):
    average_face_vector = get_average_face_vector(face_matrix)

    # Centers the data by subtracting the average_face_vector from each row.
    centered_face_matrix = face_matrix - average_face_vector

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

def main():
    olivetti_faces = fetch_olivetti_faces() # Gets data set with 400 64x64 images of faces in greyscale.
    face_matrix  = olivetti_faces['data'] # 400x(64 * 64) matrix that contains the data for the faces.
    labels = olivetti_faces['target']

    eigenfaces = get_eigenfaces(face_matrix, 0.95)

    # DEMO
    # Visualizes the average face.
    average_face = get_average_face_vector(face_matrix)
    visualize_face(get_average_face_vector(face_matrix))

    # Visualizes one of the eigenfaces.
    visualize_face(eigenfaces[:, 0])

    # Gets the weights of the first face, in other words the multidimensional point that represents the first face.
    first_face = face_matrix[0, :]
    first_face_centered = first_face - average_face
    weights = np.matmul(first_face_centered, eigenfaces)

    # Using the obtained weights, the image of the first face in the data set is recreated, and can be compared with the original image.
    first_face_recreated = average_face.reshape(-1, 1) + np.dot(eigenfaces, weights.reshape(-1, 1))
    visualize_face(first_face_recreated)
    visualize_face(first_face)

if __name__ == "__main__":
    main()
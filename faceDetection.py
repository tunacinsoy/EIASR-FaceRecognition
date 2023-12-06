from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def detect_edges(image):
    # Convert to grayscale
    gray_image = image.convert('L')
    
    # Define Sobel operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Convert to numpy array
    gray_array = np.array(gray_image)
    
    # Apply Sobel filters
    edge_x = np.abs(convolve2d(gray_array, sobel_x, boundary='symm', mode='same'))
    edge_y = np.abs(convolve2d(gray_array, sobel_y, boundary='symm', mode='same'))
    
    # Combine the edges
    edge_array = np.hypot(edge_x, edge_y)
    edge_array *= 255.0 / np.max(edge_array)
    
    # Convert back to image
    edge_image = Image.fromarray(edge_array.astype('uint8'))
    
    return edge_image

# Load your image
image_path = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\GeorgeBush\\train\\George_W_Bush_0006.jpg"
image = Image.open(image_path)

# Detect edges
edge_detected_image = detect_edges(image)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_detected_image, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')

plt.show()

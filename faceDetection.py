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

def apply_threshold(edge_image, threshold_value=50):
    # Convert edge image to numpy array if it's not already
    edge_array = np.array(edge_image)
    
    # Apply threshold
    binary_image = np.where(edge_array > threshold_value, 255, 0).astype('uint8')

    # Convert back to PIL image
    binary_image = Image.fromarray(binary_image)
    
    return binary_image

# Load your image
image_path = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\GeorgeBush\\train\\George_W_Bush_0006.jpg"
image = Image.open(image_path)

# Detect edges
edge_detected_image = detect_edges(image)

# Apply threshold to the edge-detected image
threshold_value = 30  # We might need to adjust this value based on your image
binary_image = apply_threshold(edge_detected_image, threshold_value)

# Display the results
plt.figure(figsize=(15, 7))  # Adjust the figure size as needed

# Subplot for the original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Subplot for the edge-detected image
plt.subplot(1, 3, 2)
plt.imshow(edge_detected_image, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')

# Subplot for the binary image after thresholding
plt.subplot(1, 3, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image After Thresholding')
plt.axis('off')

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.show()


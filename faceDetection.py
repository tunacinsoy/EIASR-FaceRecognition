from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import binary_dilation, binary_erosion
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

def apply_threshold(edge_array, threshold_value=30):
   # Apply threshold on the numpy array
    binary_array = np.where(edge_array > threshold_value, 1, 0).astype('uint8')  # Binary image has values 0 and 1
    
    return binary_array

def morphological_operations(binary_array, iterations=2):
    # Apply dilation
    dilated_image = binary_dilation(binary_array, iterations=iterations)
    # Apply erosion
    eroded_image = binary_erosion(dilated_image, iterations=iterations)
    
    return eroded_image

# Load your image
image_path = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\GeorgeBush\\train\\George_W_Bush_0006.jpg"
image = Image.open(image_path)

# Detect edges and convert to numpy array for processing
edge_detected_array = np.array(detect_edges(image))

# Apply threshold to the edge-detected numpy array
binary_array = apply_threshold(edge_detected_array, threshold_value=30)

# Apply morphological operations to the binary numpy array
morph_array = morphological_operations(binary_array, iterations=1)

# Convert numpy arrays back to PIL images for display
binary_image = Image.fromarray(binary_array * 255)  # Scale binary image back to [0, 255]
morph_image = Image.fromarray(morph_array * 255)  # Scale morphological image back to [0, 255]

# Display the results
plt.figure(figsize=(20, 10))

# Original image
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Edge Detected Image
plt.subplot(1, 4, 2)
plt.imshow(edge_detected_array, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')

# Binary Image After Thresholding
plt.subplot(1, 4, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image After Thresholding')
plt.axis('off')

# Morphological Operations
plt.subplot(1, 4, 4)
plt.imshow(morph_image, cmap='gray')
plt.title('Morphological Operations')
plt.axis('off')

plt.tight_layout()
plt.show()


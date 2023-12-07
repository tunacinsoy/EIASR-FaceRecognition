from PIL import Image, ImageOps
import numpy as np
from skimage.measure import regionprops
from scipy.signal import convolve2d
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import label
import matplotlib.pyplot as plt
import os

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

def segment_face(binary_array):
    # Label different regions in the binary image
    labeled_array, num_features = label(binary_array)
    
    # Find the largest labeled region, as this is most likely to be the face
    largest_region = 0
    largest_area = 0
    for region in range(1, num_features + 1):  # Start from 1 as 0 is the background
        area = np.sum(labeled_array == region)
        if area > largest_area:
            largest_area = area
            largest_region = region
    
    # Create a mask for the largest region
    face_mask = (labeled_array == largest_region)

    return face_mask

def crop_and_resize_face(original_image, face_mask, output_size=(64, 64)):
    # Label the image to separate different regions
    labeled_array, num_features = label(face_mask)
    # Get properties of labeled regions
    regions = regionprops(labeled_array)
    
    if not regions:
        print("No regions found. Returning None")
        return None
    
    # Find the bounding box of the largest region, assumed to be the face
    region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox

    # Calculate the height and width of the bounding box
    height = maxr - minr
    width = maxc - minc
    
    # Adjust the bounding box to exclude the neck and refine the top of the head
    # These ratios might need to be adjusted for your particular images
    vertical_reduce_ratio = 0.20  # Reduce from both top and bottom
    horizontal_reduce_ratio = 0.25  # Reduce from both sides
    new_minr = int(minr + height * vertical_reduce_ratio)
    new_maxr = int(maxr - height * vertical_reduce_ratio)
    new_minc = int(minc + width * horizontal_reduce_ratio)
    new_maxc = int(maxc - width * horizontal_reduce_ratio)

    # Crop the original image to the new face region using the adjusted bounding box
    face_region = original_image.crop((new_minc, new_minr, new_maxc, new_maxr))
    
    # Resize the cropped face to the desired output size
    resized_face = face_region.resize(output_size, Image.LANCZOS)
    
    # Convert to grayscale to ensure it's in black and white
    resized_face_bw = resized_face.convert('L')
    
    return resized_face_bw

# Load your image
image_path = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\GeorgeBush\\train\\George_W_Bush_0035.jpg"
image = Image.open(image_path)

# Detect edges and convert to numpy array for processing
edge_detected_array = np.array(detect_edges(image))

# Apply threshold to the edge-detected numpy array
binary_array = apply_threshold(edge_detected_array, threshold_value=30)

# Apply morphological operations to the binary numpy array
morph_array = morphological_operations(binary_array, iterations=2)

# Convert numpy arrays back to PIL images for display
binary_image = Image.fromarray(binary_array * 255)  # Scale binary image back to [0, 255]
morph_image = Image.fromarray(morph_array * 255)  # Scale morphological image back to [0, 255]

# Apply the segmentation to find the face region
face_mask = segment_face(morph_array)

# Convert the face mask back to PIL image for display
face_mask_image = Image.fromarray((face_mask * 255).astype('uint8'))

# Assuming the original image is already loaded as `image`
# and `face_mask` is the binary mask of the face region
resized_face_bw = crop_and_resize_face(image, face_mask)

# Display the results
plt.figure(figsize=(25, 10))  # Adjust the figure size as needed

# Original image
plt.subplot(1, 6, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Edge Detected Image
plt.subplot(1, 6, 2)
plt.imshow(edge_detected_array, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')

# Binary Image After Thresholding
plt.subplot(1, 6, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image After Thresholding')
plt.axis('off')

# Morphological Operations
plt.subplot(1, 6, 4)
plt.imshow(morph_image, cmap='gray')
plt.title('Morphological Operations')
plt.axis('off')

# Segmented Face Region
plt.subplot(1, 6, 5)
plt.imshow(face_mask_image, cmap='gray')
plt.title('Segmented Face Region')
plt.axis('off')

# Resized Face (black and white)
plt.subplot(1, 6, 6)
plt.imshow(resized_face_bw, cmap='gray')
plt.title('Resized Face')
plt.axis('off')

plt.tight_layout()
plt.show()

#Saving the output image

# Assuming `image_path` holds the path to the original image
# Extract the base name without the extension
base_name = os.path.basename(image_path)
name, ext = os.path.splitext(base_name)

# Create the new name by appending "_bw" to the original name
new_name = "{}_bw{}".format(name, ext)

# Specify the directory where you want to save the new image
output_directory = "C:\\Users\\tcins\\vscode-workspace\\outputs"
new_image_path = os.path.join(output_directory, new_name)

# Save the image using the new path
resized_face_bw.save(new_image_path)

print(f"Image saved as {new_image_path}")

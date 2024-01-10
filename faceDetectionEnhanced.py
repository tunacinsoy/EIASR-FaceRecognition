from PIL import Image, ImageOps
import numpy as np
from skimage.measure import regionprops, label
from skimage.feature import canny
from skimage.filters import threshold_local
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
from scipy.ndimage import label
import matplotlib.pyplot as plt
import os
import glob
import cv2

def detect_edges(image):
    # Convert to grayscale
    gray_image = image.convert('L')
    gray_array = np.array(gray_image)

    # Apply Canny edge detector
    edges = canny(gray_array, sigma=1)

    # Convert back to image
    edge_image = Image.fromarray(edges.astype('uint8') * 255)

    return edge_image

def skin_color_segmentation(image):
    # Convert to YCbCr color space
    ycbcr_image = image.convert('YCbCr')
    ycbcr_array = np.array(ycbcr_image)

    # Define skin color range in YCbCr
    min_YCbCr = np.array([0, 133, 77], np.uint8)
    max_YCbCr = np.array([235, 173, 127], np.uint8)

    # Create a mask for skin color
    skin_mask = cv2.inRange(ycbcr_array, min_YCbCr, max_YCbCr)

    return skin_mask

def equalize_histogram(image):
    gray_image = image.convert('L')
    equalized_image = ImageOps.equalize(gray_image)
    return equalized_image

def apply_threshold(edge_array, block_size=35, C=5):
    # Adaptive thresholding
    thresh = threshold_local(edge_array, block_size, offset=C)
    binary_array = (edge_array > thresh).astype('uint8')

    return binary_array

def morphological_operations(binary_array, iterations=2):
    # Apply dilation
    dilated_image = binary_dilation(binary_array, iterations=iterations)
    # Apply erosion
    eroded_image = binary_erosion(dilated_image, iterations=iterations)
    # Apply closing
    closed_image = binary_closing(eroded_image, iterations=iterations)

    return closed_image

def segment_face(binary_array):
    # Label different regions in the binary image
    labeled_array, num_features = label(binary_array)

    # Candidate regions for the face
    candidate_regions = []
    
    for region in regionprops(labeled_array):
        # Calculate the aspect ratio of the bounding box
        min_row, min_col, max_row, max_col = region.bbox
        height = max_row - min_row
        width = max_col - min_col
        aspect_ratio = width / height if height > 0 else 0

        # Check the aspect ratio
        if 0.75 < aspect_ratio < 1.3:  # Assuming face aspect ratio
            candidate_regions.append(region)

    # Choose the region closest to the center of the image
    # Assuming largest area is the face
    face_region = max(candidate_regions, key=lambda r: r.area) if candidate_regions else None

    # Create a mask for the face region, if a region is found
    face_mask = (labeled_array == face_region.label) if face_region else None

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
    # These ratios might need to be adjusted
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

# Directory path with a pattern to match all images
pattern = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\subjects\\Vladimir_Putin\\*.jpg"

for image_path in glob.glob(pattern):

    # Load the image
    image = Image.open(image_path)

    # Apply histogram equalization
    equalized_image = equalize_histogram(image)

    # Continue with skin color segmentation (if you've added this step)
    skin_mask = skin_color_segmentation(equalized_image)

    # Convert skin mask to PIL Image for further processing (if using skin color segmentation)
    skin_mask_image = Image.fromarray(skin_mask)

    # Proceed with edge detection
    # Note: You can decide whether to use the original or the skin-masked image here
    edge_detected_array = np.array(detect_edges(skin_mask_image))  # or equalized_image

    # Apply threshold to the edge-detected numpy array
    binary_array = apply_threshold(edge_detected_array, block_size=35, C=5)

    # Apply morphological operations to the binary numpy array
    morph_array = morphological_operations(binary_array, iterations=2)

    # Convert numpy arrays back to PIL images for display
    binary_image = Image.fromarray(binary_array * 255)  # Scale binary image back to [0, 255]
    morph_image = Image.fromarray(morph_array * 255)  # Scale morphological image back to [0, 255]

    # Apply the segmentation to find the face region
    face_mask = segment_face(morph_array)

    # Convert the face mask back to PIL image for display
    face_mask_image = Image.fromarray((face_mask * 255).astype('uint8'))

    # and `face_mask` is the binary mask of the face region
    resized_face_bw = crop_and_resize_face(image, face_mask)

    #Saving the output image

    # Extract the base name without the extension
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    # Create the new name by appending "_bw" to the original name
    new_name = "{}_bw{}".format(name, ext)

    # Specify the directory where you want to save the new image
    output_directory = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\outputsOfEnhanced\\Vladimir_Putin"
    new_image_path = os.path.join(output_directory, new_name)

    # Save the image using the new path
    resized_face_bw.save(new_image_path)

    print(f"Image saved as {new_image_path}")

# # Display the results in a grid with 2 rows and 3 columns
# plt.figure(figsize=(15, 10)) 

# # Original image
# plt.subplot(2, 3, 1)
# plt.imshow(image)
# plt.title('1. Original Image')
# plt.axis('off')

# # Edge Detected Image
# plt.subplot(2, 3, 2)
# plt.imshow(edge_detected_array, cmap='gray')
# plt.title('2. Edge Detected Image')
# plt.axis('off')

# # Binary Image After Thresholding
# plt.subplot(2, 3, 3)
# plt.imshow(binary_image, cmap='gray')
# plt.title('3. Binary Image After Thresholding')
# plt.axis('off')

# # Morphological Operations
# plt.subplot(2, 3, 4)
# plt.imshow(morph_image, cmap='gray')
# plt.title('4. Morphological Operations')
# plt.axis('off')

# # Segmented Face Region
# plt.subplot(2, 3, 5)
# plt.imshow(face_mask_image, cmap='gray')
# plt.title('5. Segmented Face Region')
# plt.axis('off')

# # Resized Face (black and white)
# plt.subplot(2, 3, 6)
# plt.imshow(resized_face_bw, cmap='gray')
# plt.title('6. Resized Face')
# plt.axis('off')

# plt.tight_layout()
# plt.show()



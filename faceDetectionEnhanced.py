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
    # Conversion to grayscale
    gray_image = image.convert('L')
    gray_array = np.array(gray_image)

    # Applying Canny edge detector
    edges = canny(gray_array, sigma=1)

    # Convert back to image
    edge_image = Image.fromarray(edges.astype('uint8') * 255)

    return edge_image

def detect_skin(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin colors
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Bitwise-AND mask and original image
    skin = cv2.bitwise_and(image, image, mask=mask)

    return skin

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
    
    # Finding the bounding box of the largest region, assumed to be the face
    region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox

    # Calculating the height and width of the bounding box
    height = maxr - minr
    width = maxc - minc
    
    # Adjusting the bounding box to exclude the neck and refine the top of the head
    vertical_reduce_ratio = 0.20  # Reduce from both top and bottom
    horizontal_reduce_ratio = 0.25  # Reduce from both sides
    new_minr = int(minr + height * vertical_reduce_ratio)
    new_maxr = int(maxr - height * vertical_reduce_ratio)
    new_minc = int(minc + width * horizontal_reduce_ratio)
    new_maxc = int(maxc - width * horizontal_reduce_ratio)

    # Cropping the original image to the new face region using the adjusted bounding box
    face_region = original_image.crop((new_minc, new_minr, new_maxc, new_maxr))
    
    # Resizing the cropped face to the desired output size
    resized_face = face_region.resize(output_size, Image.LANCZOS)
    
    # Converting to grayscale to ensure it's in black and white
    resized_face_bw = resized_face.convert('L')
    
    return resized_face_bw

def display_image(image, title, subplot=None):
    if subplot is None:
        plt.imshow(image, cmap='gray')
    else:
        plt.subplot(*subplot)
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Directory path with a pattern to match all images
pattern = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\subjects\\Kofi_Annan\\*.jpg"

for image_path in glob.glob(pattern):

    # Loading the image
    image = Image.open(image_path)

    # Display original image
    display_image(np.array(image), "Original Image", (3, 3, 1))

    # Starting with skin color segmentation
    image2 = cv2.imread(image_path)
    skin_mask = detect_skin(image2)

    # Converting skin mask to PIL Image for further processing
    skin_mask_image = Image.fromarray(skin_mask)
    display_image(skin_mask, "Skin Mask", (3, 3, 2))

    # Applying histogram equalization
    equalized_image = equalize_histogram(skin_mask_image)
    display_image(np.array(equalized_image), "Equalized Image", (3, 3, 3))

    # Proceeding with edge detection
    edge_detected_array = np.array(detect_edges(equalized_image))
    display_image(edge_detected_array, "Edge Detection", (3, 3, 4))

    # Applying threshold to the edge-detected numpy array
    binary_array = apply_threshold(edge_detected_array, block_size=35, C=5)
    binary_image = Image.fromarray(binary_array * 255)
    display_image(binary_array, "Threshold Applied", (3, 3, 5))

    # Applying morphological operations to the binary numpy array
    morph_array = morphological_operations(binary_array, iterations=2)
    morph_image = Image.fromarray(morph_array * 255)
    display_image(morph_array, "Morphological Operations", (3, 3, 6))

    # Applying the segmentation to find the face regions
    possible_face_regions = segment_face(morph_array)
    face_mask_image = Image.fromarray((possible_face_regions * 255).astype('uint8'))
    display_image(possible_face_regions, "Face Mask", (3, 3, 7))

    # Resizing and cropping the image
    resized_face_bw = crop_and_resize_face(image, possible_face_regions)
    display_image(np.array(resized_face_bw), "Cropped and Resized Face", (3, 3, 8))

    # Saving and showing the results
    plt.show()
    
    # Saving the output image

    # Extract the base name without the extension
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    # Creating the new name by appending "_bw" to the original name
    new_name = "{}_bw{}".format(name, ext)

    # Specifying the directory where you want to save the new image
    output_directory = "C:\\Users\\tcins\\vscode-workspace\\EIASR-FaceRecognition\\outputsOfEnhanced\\Kofi_Annan"
    new_image_path = os.path.join(output_directory, new_name)

    # Saving the image using the new path
    resized_face_bw.save(new_image_path)

    print(f"Image saved as {new_image_path}")



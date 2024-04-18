from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from skimage import io
from skimage.color import rgb2gray
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance


# ---------- Create Variables ----------

# Assining baby.tiff image to a variable
tiff = io.imread('data/baby.tiff')

#Assign width, height, depth to variables
img_width = tiff.shape[1]
img_height = tiff.shape[0]
img_depth = tiff.dtype.itemsize * 8

# Print Values
print(f"Width: {img_width}")
print(f"Height: {img_height}")
print(f"Bit Depth: {img_depth} bits per pixel")

# Converting function into double-precision array
double_precision_array = tiff.astype(np.float64)
print(f"Double-precision array:]\n{double_precision_array}\n")


# ---------- Linearization ----------
black = 0
white = 16383

# Apply Linearization
linearized_img = (double_precision_array - black) / (white - black)

# Create bounds by clipping values
linearized_img = np.clip(linearized_img, 0, 1)

# Print values
print(f"Image after being clipped and linearized:\n{linearized_img}\n")

# ---------- Bayer Pattern

def get_bayer_pattern(image):
    # Getting top-left 2 x 2
    top_left_corner = image[:2, :2]

    # Getting mean values
    mean_values = np.mean(top_left_corner, axis= (0, 1))

    # Analyize all possible bayer patterns and choose the closest match
    possible_bayer_patterns = ['grbg', 'rggb', 'bggr', 'gbrg']
    best_bayer_patterns = np.argmin(np.abs(np.array(mean_values) - [128, 128, 128]))
    plt.imsave('top_left_corner.png', top_left_corner)
    # Return value
    return possible_bayer_patterns[best_bayer_patterns]

# Get bayer pattern
bayer_pattern = get_bayer_pattern(linearized_img)

print(f"Bayer Pattern: \n{bayer_pattern}\n")


# ---------- White Balancing ----------

# Values from dcraw
r_scale = 1.628906
g_scale = 1.000000
b_scale = 1.386719

def gray_bal(image):
    return image / np.mean(image)

def white_bal(image):
    return image / np.mean(image, axis=(0, 1))

# Function to create campera presets
def camera_presets(image, r_scale, g_scale, b_scale):
    reshaped_img = image.reshape(-1, 3)

    for i in range(0, len(reshaped_img)):
        if i % 4 == 0:
            reshaped_img[i] *= r_scale  
        elif i % 4 == 1 or i % 4 == 2:
            reshaped_img[i] *= g_scale  
        elif i % 4 == 3:
            reshaped_img[i] *= b_scale  
    
    balanced_img = reshaped_img.reshape(image.shape)
    return balanced_img

white_balanced_img = white_bal(linearized_img)
gray_balanced_img = gray_bal(linearized_img)
camera_presets_img = camera_presets(linearized_img, r_scale, g_scale, b_scale)


# Used imsave to get top left of image, top right and bottom left were the same.
# The bayer patter could either be rggb or bggr, but rggb matches colors correctly


# ---------- Demosaicing ----------

def demosaic_image(image, pattern):
    # Identify pattern based on above bayer pattern
    # Could not find a better way to do this
    return demosaicing_CFA_Bayer_bilinear(image, pattern)
       
# Print demosiac images
print(f"Demosaiced White Balanced Image:\n{demosaic_image(white_balanced_img, 'rggb')}\n")
print(f"Demosaiced Gray Balanced Image:\n{demosaic_image(gray_balanced_img, 'rggb')}\n")
print(f"Demosaiced Camera Presets Image:\n{demosaic_image(camera_presets_img, 'rggb')}\n")


# ---------- Color Space Correction ----------
def color_space_correction(image, mxyz_to_camera):
    # Transform array to xyz image
    MsRGB_to_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    M_XYZ_To_Camera = mxyz_to_camera / 10000.0

    M_XYZ_To_Camera = M_XYZ_To_Camera.reshape(3,3)

    # Compute sRGB to camera specific colour
    MsRGB_to_cam = np.dot(M_XYZ_To_Camera, MsRGB_to_XYZ)

    # Normalize matrix
    MsRGB_to_cam /= MsRGB_to_cam.sum(axis=1)[:, np.newaxis]

    # Compute Inverse
    M_cam_to_sRGB = np.linalg.inv(MsRGB_to_cam)

    # Apply color correction
    corrected_img = np.dot(image.reshape(-1, 3), M_cam_to_sRGB.T).reshape(image.shape)

    # Clip values to appropiate values
    corrected_img = np.clip(corrected_img, 0, 1)

    return corrected_img

# From dcraw source code, found camera to be Nikon D3400
# {6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318}
camera_matrix = np.array([6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318])
white_balanced_img_sRGB = color_space_correction(demosaic_image(white_balanced_img, 'rggb'), camera_matrix)
gray_balanced_img_sRGB = color_space_correction(demosaic_image(gray_balanced_img, 'rggb'), camera_matrix)
camera_balanced_img_sRGB = color_space_correction(demosaic_image(camera_presets_img, 'rggb'), camera_matrix)


print(f"White balanced image, corrected color space:\n{white_balanced_img_sRGB}\n")
print(f"Gray balanced image, corrected color space:\n{gray_balanced_img_sRGB}\n")
print(f"Camera presets image corrected color space:\n{camera_balanced_img_sRGB}\n")


# ---------- Brightness adjustment and gamma encoding ----------

# Gamma Encoding
# Chose mean = 0.2
def gamma_encode(image, mean = 0.2):
    grayscale = rgb2gray(image)
    mean_intensity = np.mean(grayscale)
    scaled_img = image * (mean / mean_intensity)
    clipped_img = np.clip(scaled_img, 0, 1)

    gamma_encoded_img = np.where(clipped_img <= .0031308, 12.92 * clipped_img, (1 + .055) * np.power(clipped_img, 1 / 2.4) - .055)

    gamma_encoded_img = np.dstack( [gamma_encoded_img[:, :, 0],
                                      gamma_encoded_img[:, :, 1],
                                      gamma_encoded_img[:, :, 2]])
    
    return gamma_encoded_img
    
gamma_encoded_white_balanced = gamma_encode(white_balanced_img_sRGB)
gamma_encoded_gray_balanced = gamma_encode(gray_balanced_img_sRGB)
gamma_encoded_camera_balanced = gamma_encode(camera_balanced_img_sRGB)


# ---------- Compression ----------

def compress(gamma_encoded_img, output_directory):

    png_file = os.path.join(output_directory, 'image.png')
    plt.imsave(png_file, gamma_encoded_img, cmap='gray')
    png_size = os.path.getsize(png_file)

    jpeg_file_q95 = os.path.join(output_directory, 'image_q95.jpeg')
    plt.imsave(jpeg_file_q95, gamma_encoded_img, cmap='gray')
    jpeg_size_q95 = os.path.getsize(jpeg_file_q95)

    compression_ratio_q95 = png_size / jpeg_size_q95
    print(f"JPEG compression ratio: {compression_ratio_q95:.2f}")

    # Find the lowest JPEG quality setting with indistinguishable quality
    for quality in range(100, 0, -5):
        jpeg_file = os.path.join(output_directory, f'image_q{quality}.jpeg')
        plt.imsave(jpeg_file, gamma_encoded_img, cmap='gray')
        jpeg_size = os.path.getsize(jpeg_file)

        if jpeg_size < png_size:
            compression_ratio = png_size / jpeg_size
            print(f"Compression ratio for JPEG with quality {quality}: {compression_ratio:.2f}")
            break


compress(gamma_encoded_white_balanced, 'data/white_balanced')
compress(gamma_encoded_gray_balanced, 'data/gray_balanced')
compress(gamma_encoded_camera_balanced, 'data/camera_presets')


# ---------- Manual White Balancing ----------
def manual_white_balance(image_path, coords, brightness_scale = 0.7):
    
    image = io.imread(image_path).astype(np.float64)
    
    # Normalize the image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Get white patch from coords
    x, y, width, height = coords
    white_patch = image[y:y+height, x:x+width]
    
    # Calculate average of RGB values in coords selected
    mean_white_values = np.mean(white_patch, axis=(0, 1))
    
    # Calculate scaling factors to normalize the RGB channels
    scale_factors = mean_white_values.max() / mean_white_values
    
    # Apply the scaling factors to the entire image
    white_balanced_img = (image * scale_factors).clip(0, 1)

    white_balanced_img *= brightness_scale

    name = os.path.basename(os.path.dirname(image_path))
    
    # Save the original image
    original_image_save_path = os.path.join('data/manual_white_balancing', f'{name}_original.png')
    plt.imsave(original_image_save_path, image)
    
    # Save the white-balanced image
    white_balanced_image_save_path = os.path.join('data/manual_white_balancing', f'{name}_white_balanced.png')
    plt.imsave(white_balanced_image_save_path, white_balanced_img)


def get_coords(image_path):
    image = io.imread(image_path)
    
    # Convert the image to double-precision array
    image = image.astype(np.float64)
    
    # Normalize the image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Display the image for white patch selection
    plt.imshow(image)
    plt.title('White Balancing - click twice to create rectangle of white pixels')
    plt.axis('on')
    
    # Use ginput to select the white patch
    points = plt.ginput(2)
    plt.close()
    
    # Calculate coordinates and size of the white patch
    x1, y1 = points[0]
    x2, y2 = points[1]
    x, y = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)
    coords = (int(x), int(y), int(width), int(height))
    return coords
 
white_patch_coords = get_coords('data/baby.jpeg')
manual_white_balance('data/white_balanced/image.png', white_patch_coords)
manual_white_balance('data/gray_balanced/image.png', white_patch_coords)
manual_white_balance('data/camera_presets/image.png', white_patch_coords)
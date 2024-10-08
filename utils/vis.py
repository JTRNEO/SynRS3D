import cv2
import numpy as np
import torch

def convert_dsm(image, max_value=None):
    # Find the maximum value in the image
    if max_value==None:
        max_value = np.max(image)
        
    image[image>max_value] = max_value
    # Invert the image
    inverted_img = max_value - image

    # Normalize the inverted image to 0-1 range
    normalized_img = cv2.normalize(inverted_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Apply a colormap for better visualization
    colormap_img = cv2.applyColorMap((normalized_img * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Convert image to CHW format
    chw_image = colormap_img.transpose(2, 0, 1)

    # Convert to tensor and normalize to [0, 1]
    tensor_image = torch.from_numpy(chw_image).float() / 255.0

    return tensor_image, max_value

def create_custom_colormap():
    color_map = {
        0: [148, 148, 148],  # Grey
        1: [34, 97, 38],     # Dark Green
        2: [222, 31, 7],     # Bright Red
        3: [255, 206, 0],    # Yellow
        4: [0, 165, 255],    # Bright Blue
        5: [255, 105, 180],  # Hot Pink
        6: [128, 0, 128],    # Purple
        7: [255, 165, 0],    # Orange
        8: [0, 255, 255],    # Cyan
        9: [112, 128, 144]   # Slate Gray
    }
    return color_map

def apply_custom_colormap(mask, color_map):
    # Initialize an empty image array with 3 channels for RGB
    h, w = mask.shape
    colormap_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in color_map.items():
        colormap_mask[mask == label] = color

    return colormap_mask

def convert_ss_mask(mask):
    color_map = create_custom_colormap()

    colormap_mask = apply_custom_colormap(mask, color_map)

    # Convert image to CHW format
    chw_mask = colormap_mask.transpose(2, 0, 1)

    # Convert to tensor and normalize to [0, 1]
    tensor_mask = torch.from_numpy(chw_mask).float() / 255.0

    return tensor_mask

def prepare_image_for_tensorboard(image_tensor):
    """
    Prepares an image tensor for visualization in TensorBoard by ensuring it is normalized and in the correct format.

    Parameters:
    - image_tensor: A PyTorch tensor of the image in the shape [1, 3, H, W].

    Returns:
    - A normalized PyTorch tensor of the image in the shape [3, H, W] with values in the range [0, 1].
    """
    # Normalize the image to [0, 1] if it's not already
    image_tensor = image_tensor.float()  # Ensure floating point for division
    min_val = float(image_tensor.min())
    max_val = float(image_tensor.max())
    normalized_image = (image_tensor - min_val) / (max_val - min_val)

    # Remove the batch dimension if present
    if normalized_image.dim() == 4 and normalized_image.shape[0] == 1:
        normalized_image = normalized_image.squeeze(0)

    return normalized_image
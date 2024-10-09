import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
import torch
import random

def get_random_indices(start_end_indices, num_indices):
    start, end = start_end_indices[0]
    return random.sample(range(start, end), num_indices)


def get_register_indices(img_embeds, text_embeds, start_end_indices):
    img_embeds, text_embeds = img_embeds.cpu(), text_embeds.cpu()

    # Get outliers
    mean = torch.mean(torch.norm(img_embeds.squeeze(), dim=-1))
    std = torch.std(torch.norm(img_embeds.squeeze(), dim=-1))
    outliers = torch.norm(img_embeds.squeeze(), dim=-1) > mean + 2 * std
    indices = torch.where(outliers)[0]

    start, end = start_end_indices[0]
    start, end = start.item(), end.item()
    actual_indices = indices + start

    return actual_indices.tolist()

def get_object_patch_indices(image, ann, start_end_indices, buffer=0):
    def _add_buffer(indices, buffer_amount=1):
        if buffer_amount <= 0:
            return set(indices)
        
        if buffer_amount > 1:
            new_indices = _add_buffer(indices, buffer_amount=buffer_amount-1)
        else:
            new_indices = set(indices)
        
        width = 24
        height = 24
        
        buffer_indices = set()
        for idx in new_indices:
            row, col = divmod(idx, width)
            
            # Check and add adjacent indices
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < height and 0 <= new_col < width:
                        buffer_indices.add(new_row * width + new_col)
        
        return new_indices.union(buffer_indices)

    image, annotations = correct_annotations_for_crop(image, ann)
    annotation = annotations[0]
    selected_annotation, overlapping_patch_indices = find_overlapping_patches(image, annotation)

    assert all([0 <= idx < 24 * 24 for idx in overlapping_patch_indices]), "Input indices must be between 0 and 575 inclusive"
    overlapping_patch_indices = _add_buffer(overlapping_patch_indices, buffer_amount=buffer)
    
    assert all([0 <= idx < 24 * 24 for idx in overlapping_patch_indices]), "Final indices must be between 0 and 575 inclusive"

    start, end = start_end_indices[0]
    start, end = start.item(), end.item()

    actual_indices = [overlapping_patch_index + start for overlapping_patch_index in overlapping_patch_indices]

    return actual_indices

def correct_annotations_for_crop(image, annotations):
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    cropped_image = image.crop((left, top, right, bottom))
    corrected_annotations = []
    if type(annotations) is not list:
        annotations = [annotations]
    for ann in annotations:
        x, y, w, h = ann['bbox']
        new_x = max(0, x - left)
        new_y = max(0, y - top)
        new_w = min(w, crop_size - new_x)
        new_h = min(h, crop_size - new_y)
        if new_w > 0 and new_h > 0:
            new_bbox = [new_x, new_y, new_w, new_h]
            new_segmentation = []
            for segment in ann['segmentation']:
                new_segment = []
                for i in range(0, len(segment), 2):
                    x = segment[i] - left
                    y = segment[i + 1] - top
                    if 0 <= x < crop_size and 0 <= y < crop_size:
                        new_segment.extend([x, y])
                if len(new_segment) >= 6:
                    new_segmentation.append(new_segment)
            if new_segmentation:
                new_ann = ann.copy()
                new_ann['bbox'] = new_bbox
                new_ann['segmentation'] = new_segmentation
                new_ann['area'] = new_w * new_h
                corrected_annotations.append(new_ann)
    return cropped_image, corrected_annotations

def find_overlapping_patches(image, selected_annotation):
    width, height = image.size
    patch_width = width // 24
    patch_height = height // 24
    segmentation = selected_annotation['segmentation']
    ann_rle = maskUtils.frPyObjects(segmentation, height, width)
    overlapping_patch_indices = []
    for i in range(24 * 24):
        patch_x = (i % 24) * patch_width
        patch_y = (i // 24) * patch_height
        patch_polygon = [
            patch_x, patch_y,
            patch_x + patch_width, patch_y,
            patch_x + patch_width, patch_y + patch_height,
            patch_x, patch_y + patch_height
        ]
        patch_rle = maskUtils.frPyObjects([patch_polygon], height, width)
        iscrowd = np.zeros(len(ann_rle), dtype=np.uint8)
        overlap = maskUtils.iou(patch_rle, ann_rle, iscrowd)[0]
        if overlap.any():
            overlapping_patch_indices.append(i)
    return selected_annotation, overlapping_patch_indices

def display_patches(image, highlight_indices=[]):
    """
    Displays an image with a 24x24 grid of patches overlaid. Optionally highlights specific patches.

    Args:
        image (PIL.Image.Image): The input image to display with patches overlaid.
        highlight_indices (list of int, optional): A list of patch indices to highlight. Default is an empty list.

    Returns:
        None
    """
    cropped_image = _crop_square(image)
    patched_image = _overlay_patches(cropped_image, highlight_indices)
    plt.figure(figsize=(12, 12))
    plt.imshow(patched_image)
    plt.axis('off')
    plt.show()

def _overlay_patches(image, highlight_indices, display_grid=True, display_numbers=True, draw_highlighted=True):
    """
    Overlays a 24x24 grid of patches on the image. Highlights specific patches if indices are provided.

    Args:
        image (PIL.Image.Image): The input image to overlay patches on.
        highlight_indices (list of int): A list of patch indices to highlight. 
            Highlighted patches are outlined in red and have thicker borders.
        display_grid (bool): Whether to draw the grid
        display_numbers (bool): Whether to draw the numbers
        display_highlighted (bool): Whether to draw the highlighted patches

    Returns:
        PIL.Image.Image: The image with the overlaid grid of patches.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    patch_size = width / 24  # 24x24 grid
    
    font = ImageFont.load_default(size=patch_size/2)

    for i in range(24):
        for j in range(24):

            x = j * patch_size
            y = i * patch_size
            patch_index = i * 24 + j
            
            color = "white" if patch_index not in highlight_indices else "red"
            width = 1 if patch_index not in highlight_indices else 3

            # Draw grid lines
            if display_grid or (patch_index in highlight_indices):
                draw.rectangle([x, y, x + patch_size, y + patch_size], outline=color, width=width)
            
            # Draw patch index
            if display_numbers:
                text = str(patch_index)
                bbox = draw.textbbox((x, y), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x + (patch_size - text_width) // 2
                text_y = y + (patch_size - text_height) // 2
                draw.text((text_x, text_y), text, fill="white", font=font)

    return image

def _crop_square(image):
    """
    Crops the input image to a square by cutting equal parts from the longer dimension.

    Args:
        image (PIL.Image.Image): The input image to crop.

    Returns:
        PIL.Image.Image: The cropped square image.
    """
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))

def replace_image_regions_with_patches(square_image, patch_indices, patch_type="noise"):
    """For a square image that is to be broken up into 24x24 patches,
    replace the patches at patch_indices with a patch.
    Works with both color and black and white images.
    """
    # Ensure the image is square
    width, height = square_image.size
    assert width == height, "Image must be square"

    # Convert PIL Image to numpy array
    image_array = np.array(square_image)
    
    # Determine if the image is grayscale or color
    is_grayscale = len(image_array.shape) == 2 or (len(image_array.shape) == 3 and image_array.shape[2] == 1)

    # Calculate the number of patches in each dimension
    image_size = width
    num_patches_per_row = 24
    patch_size = image_size // num_patches_per_row

    if patch_type == "noise":
        if is_grayscale:
            patch = np.random.rand(patch_size, patch_size) * 255  # Random grayscale values
        else:
            patch = np.random.rand(patch_size, patch_size, 3) * 255  # Random RGB values
        patch = patch.astype(np.uint8)
    elif patch_type == "gray":
        if is_grayscale:
            patch = np.ones((patch_size, patch_size), dtype=np.uint8) * 128  # Medium gray
        else:
            patch = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 128  # Medium gray
    else:
        raise ValueError("Unsupported patch_type. Use 'noise' or 'gray'.")

    for patch_index in patch_indices:
        row, col = divmod(patch_index, num_patches_per_row)

        # Calculate the pixel coordinates for the patch
        start_row = row * patch_size
        start_col = col * patch_size
        end_row = start_row + patch_size
        end_col = start_col + patch_size
        
        # Replace the region with the patch
        if is_grayscale:
            image_array[start_row:end_row, start_col:end_col] = patch
        else:
            image_array[start_row:end_row, start_col:end_col, :] = patch

    # Convert the modified numpy array back to a PIL Image
    if is_grayscale:
        modified_image = Image.fromarray(image_array, mode='L')
    else:
        modified_image = Image.fromarray(image_array, mode='RGB')
    
    return modified_image
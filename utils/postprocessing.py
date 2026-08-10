import torch
import numpy as np
from skimage import measure

def largest_connected_component(images, threshold=0.7):
    """
    Keeps only the largest connected component in either a batch of 2D images or a single 2D image.
    Args:
    - images (numpy array): Input batch of images (Batch Size, W, H) or a single image (W, H)
    - threshold (float): Pixels with intensities below this are considered background.

    Returns:
    - numpy array: Processed batch of images or a single processed image.
    """
    if isinstance(images, torch.Tensor): images = images.numpy() #convert to numpy if not

    # Check if input is a single image by its dimension
    if images.ndim == 2:
        images = images[np.newaxis, ...]  # Add a batch dimension if it's a single image

    # Initialize processed batch with the same shape and type as input
    processed_batch = np.zeros_like(images)

    for i in range(images.shape[0]):
        # Apply threshold
        binary_image = images[i] > threshold

        # Label connected components
        labels = measure.label(binary_image, connectivity=2)
        if labels.max() == 0:
            continue  # No components found

        # Largest component by pixel area (regionprops; avoids bincount/argmax edge cases)
        regions = measure.regionprops(labels)
        largest = max(regions, key=lambda r: int(r.area))
        processed_batch[i] = (labels == int(largest.label)).astype(float)

    # If it was a single image input, remove the batch dimension before returning
    if processed_batch.shape[0] == 1:
        return processed_batch[0]
    else:
        return processed_batch

import numpy as np
from skimage import measure

def extract_central_components(images, threshold=0.7, center_height=0.5, center_width=0.5):
    """
    Keeps components located in the specified central area of the image.
    
    Args:
    - images (numpy array): Input batch of images (Batch Size, H, W) or single image (H, W)
    - threshold (float): Pixels below this are considered background
    - center_height (float): Proportion of vertical center area to consider (0-1)
    - center_width (float): Proportion of horizontal center area to consider (0-1)
    
    Returns:
    - numpy array: Processed batch/images with only central components
    """
    if isinstance(images, torch.Tensor):
        images = images.numpy()

    # Validate input parameters
    if not (0 < center_height <= 1) or not (0 < center_width <= 1):
        raise ValueError("Center dimensions must be between 0 and 1")

    # Add batch dimension if single image
    if images.ndim == 2:
        images = images[np.newaxis, ...]

    processed_batch = np.zeros_like(images)

    for i in range(images.shape[0]):
        # Threshold to create binary image
        binary_image = images[i] > threshold
        
        # Label connected components
        labels = measure.label(binary_image, connectivity=2)
        if labels.max() == 0:  # No components found
            continue

        # Calculate ROI boundaries using configurable dimensions
        H, W = binary_image.shape
        y_margin = (1 - center_height) / 2
        x_margin = (1 - center_width) / 2
        
        y_start, y_end = int(round(H * y_margin)), int(round(H * (1 - y_margin)))
        x_start, x_end = int(round(W * x_margin)), int(round(W * (1 - x_margin)))

        # Create ROI mask
        roi_mask = np.zeros_like(binary_image, dtype=bool)
        roi_mask[y_start:y_end, x_start:x_end] = True

        # Find components overlapping with ROI
        overlap_labels = np.unique(labels[roi_mask])
        overlap_labels = overlap_labels[overlap_labels != 0]  # Exclude background

        # Create output mask
        if len(overlap_labels) > 0:
            component_mask = np.isin(labels, overlap_labels)
            processed_batch[i] = component_mask.astype(float)

    return processed_batch[0] if processed_batch.shape[0] == 1 else processed_batch

import numpy as np
import networkx as nx
from skimage import measure, morphology, util
from skimage.graph import route_through_array

def clean_skeleton(image):
    """
    Cleans a skeletonized image by keeping the longest continuous path, 
    removing small branches that are connected to the main path.
    
    Args:
    - image (numpy.ndarray): Input 2D skeletonized image (W, H)

    Returns:
    - numpy.ndarray: Processed image.
    """
    # Skeleton must be binary, ensure it is
    skeleton = image > 0.5

    # Convert skeleton to graph
    G = nx.Graph()
    for r, c in np.argwhere(skeleton):
        # Add edges to all 8 possible neighbors
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr != 0 or dc != 0:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < skeleton.shape[0] and 0 <= cc < skeleton.shape[1] and skeleton[rr, cc]:
                        G.add_edge((r, c), (rr, cc))

    # Find all paths from each junction or endpoint to another
    endpoints = [node for node, degree in G.degree if degree == 1]
    max_path = []
    max_length = 0

    # Compute the longest path in the graph
    for start in endpoints:
        lengths = nx.single_source_dijkstra_path_length(G, start)
        farthest_node = max(lengths, key=lengths.get)
        path_length = lengths[farthest_node]
        if path_length > max_length:
            max_path = nx.shortest_path(G, start, farthest_node)
            max_length = path_length

    # Create an image from the longest path
    cleaned_image = np.zeros_like(image)
    for r, c in max_path:
        cleaned_image[r, c] = 1

    return cleaned_image

def convert_msd_to_mm(original_shape, new_shape, msd_pixel_score):
    """
    Convert MSD score from pixels to mm considering the original and resized image shapes.
    
    Parameters:
    original_shape (tuple): Original shape of the image (height, width).
    new_shape (tuple): Resized shape of the image (height, width).
    msd_pixel_score (float): MSD score in pixels.
    
    Returns:
    float: MSD score in mm.
    """
    reference_pixel_length = 439 # only the area of the US image where the UTI is located
    reference_mm_length = 90  # in mm

    pixel_to_mm_factor = reference_mm_length / reference_pixel_length

    height_scale = original_shape[0] / new_shape[0]
    width_scale = original_shape[1] / new_shape[1]

    average_scale = (height_scale + width_scale) / 2

    adjusted_msd_pixel_score = msd_pixel_score * average_scale

    msd_mm_score = adjusted_msd_pixel_score * pixel_to_mm_factor
    
    return msd_mm_score

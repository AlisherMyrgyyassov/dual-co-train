from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.filters import gaussian
import numpy as np
from scipy import ndimage

def get_largest_connected_component(mask):
    """
    Extract the largest connected component from a binary mask.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask
    
    Returns:
    --------
    largest_component : numpy.ndarray
        Binary mask with only the largest connected component
    """
    # Label connected components
    labeled_mask, num_features = label(mask > 0.5)
    
    if num_features == 0:
        return mask
    
    # Find the largest component
    component_sizes = []
    for i in range(1, num_features + 1):
        component_sizes.append((labeled_mask == i).sum())
    
    largest_component_id = np.argmax(component_sizes) + 1
    largest_component = (labeled_mask == largest_component_id).astype(np.uint8)
    
    return largest_component

def smooth_mask_edges(mask, smoothing_iterations=2, gaussian_sigma=1.0, morphological_size=3, preserve_connectivity=True):
    """
    Smooth mask edges to prepare for skeletonization.
    Uses a combination of morphological operations and Gaussian smoothing.
    Optionally preserves connectivity to prevent splitting connected components.
    
    Default Parameter Rationale (optimized for 224x224 ultrasound images):
    ----------------------------------------------------------------------
    - smoothing_iterations=2: Balance between smoothing quality and detail preservation.
      One iteration insufficient, three+ may over-smooth anatomical features.
    
    - gaussian_sigma=1.0: ~1 pixel radius smoothing (~0.45% of 224px width). Appropriate
      for removing segmentation artifacts without blurring muscle boundaries.
    
    - morphological_size=3: 7x7 structuring element removes 1-3 pixel artifacts while
      preserving muscle regions (typically >10 pixels wide).
    
    - preserve_connectivity=True: Critical for single-component muscle masks. Prevents
      splitting during morphological operations using adaptive thresholding (0.35) and
      reconnection strategies.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask
    smoothing_iterations : int
        Number of smoothing iterations (default: 2)
    gaussian_sigma : float
        Sigma for Gaussian smoothing (default: 1.0)
    morphological_size : int
        Size of morphological operations (default: 3)
    preserve_connectivity : bool
        If True, ensures the mask remains connected after smoothing.
        Uses a lower threshold after Gaussian smoothing and reconnects if split.
        (default: True)
    
    Returns:
    --------
    smoothed_mask : numpy.ndarray
        Smoothed binary mask
    """
    smoothed = mask.astype(np.float32)
    
    # Get initial number of connected components
    if preserve_connectivity:
        initial_labeled, initial_num_components = label(smoothed > 0.5)
        preserve_single_component = (initial_num_components == 1)
    else:
        preserve_single_component = False
    
    for _ in range(smoothing_iterations):
        # Fill small holes
        smoothed = binary_fill_holes(smoothed > 0.5).astype(np.float32)
        
        # Morphological closing to fill small gaps
        smoothed = binary_closing(smoothed > 0.5, disk(morphological_size)).astype(np.float32)
        
        # Morphological opening to remove small protrusions
        # Only apply if it doesn't break connectivity (for single-component masks)
        if preserve_single_component:
            # Check connectivity before opening
            before_opening_labeled, before_opening_num = label(smoothed > 0.5)
            
            opened = binary_opening(smoothed > 0.5, disk(morphological_size)).astype(np.float32)
            opened_labeled, opened_num = label(opened > 0.5)
            
            # If opening breaks connectivity, use a more conservative approach
            if opened_num > 1 and before_opening_num == 1:
                # Try with smaller kernel
                smaller_size = max(1, morphological_size - 1)
                opened_small = binary_opening(smoothed > 0.5, disk(smaller_size)).astype(np.float32)
                opened_small_labeled, opened_small_num = label(opened_small > 0.5)
                
                if opened_small_num == 1:
                    smoothed = opened_small
                else:
                    # Skip opening to preserve connectivity
                    smoothed = smoothed
            else:
                smoothed = opened
        else:
            smoothed = binary_opening(smoothed > 0.5, disk(morphological_size)).astype(np.float32)
        
        # Gaussian smoothing for edge smoothing
        smoothed = gaussian(smoothed, sigma=gaussian_sigma)
        
        # Use a lower threshold to preserve thin connections
        # This helps maintain connectivity after Gaussian smoothing
        if preserve_single_component:
            # Lower threshold preserves thin bridges
            threshold = 0.35
        else:
            threshold = 0.5
        
        smoothed = (smoothed > threshold).astype(np.float32)
        
        # Reconnect if connectivity was broken (for single-component masks)
        if preserve_single_component:
            smoothed_labeled, smoothed_num = label(smoothed > 0.5)
            if smoothed_num > 1:
                # Find the largest component
                component_sizes = []
                for i in range(1, smoothed_num + 1):
                    component_sizes.append((smoothed_labeled == i).sum())
                largest_id = np.argmax(component_sizes) + 1
                
                # Get largest component
                largest_component = (smoothed_labeled == largest_id).astype(np.float32)
                
                # Use morphological closing to reconnect nearby components
                # This is a gentle approach that tries to bridge small gaps
                reconnected = binary_closing(largest_component > 0.5, disk(morphological_size)).astype(np.float32)
                reconnected_labeled, reconnected_num = label(reconnected > 0.5)
                
                if reconnected_num == 1:
                    smoothed = reconnected
                else:
                    # If still disconnected, use the largest component
                    # This prevents splitting but may lose some edge detail
                    smoothed = largest_component
    
    return smoothed.astype(np.uint8)

def calculate_thickness(skeleton, mask, middle_only=False, middle_range=None):
    """
    Calculate thickness statistics as the distance from skeleton to mask edges.
    Optionally calculates thickness only for the middle portion (25% to 75%).
    
    Parameters:
    -----------
    skeleton : numpy.ndarray
        Skeleton of the mask
    mask : numpy.ndarray
        Binary mask
    middle_only : bool
        If True, only calculate thickness for middle 50% (25%-75%) of skeleton
    middle_range : tuple, optional
        Custom middle range as (start_percent, end_percent), e.g., (0.45, 0.55) for 10% middle
        If provided, overrides middle_only
    
    Returns:
    --------
    thickness_stats : dict
        Dictionary containing:
        - 'mean': mean thickness in pixels
        - 'median': median thickness in pixels
        - 'max': maximum thickness in pixels
        - 'std': standard deviation of thickness in pixels
    distance_map : numpy.ndarray
        Distance map from skeleton to edges
    """
    # Create distance transform from mask boundary
    # This gives distance from each point to the nearest edge
    mask_dist = ndimage.distance_transform_edt(mask > 0)
    
    # For each skeleton point, find the distance to the nearest edge
    skeleton_points = np.where(skeleton > 0)
    if len(skeleton_points[0]) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'max': 0.0,
            'std': 0.0
        }, mask_dist
    
    skeleton_y = skeleton_points[0]
    skeleton_x = skeleton_points[1]
    
    if middle_range is not None:
        # Custom middle range specified
        start_pct, end_pct = middle_range
        # Sort skeleton points by x-coordinate (left to right)
        sorted_indices = np.argsort(skeleton_x)
        sorted_x = skeleton_x[sorted_indices]
        sorted_y = skeleton_y[sorted_indices]
        
        # Get custom middle range
        n_points = len(sorted_x)
        start_idx = int(n_points * start_pct)
        end_idx = int(n_points * end_pct)
        
        if end_idx <= start_idx:
            # Not enough points, use all
            middle_x = sorted_x
            middle_y = sorted_y
        else:
            middle_x = sorted_x[start_idx:end_idx]
            middle_y = sorted_y[start_idx:end_idx]
        
        # Get distances at middle skeleton points
        distances = mask_dist[middle_y, middle_x]
    elif middle_only:
        # Sort skeleton points by x-coordinate (left to right)
        sorted_indices = np.argsort(skeleton_x)
        sorted_x = skeleton_x[sorted_indices]
        sorted_y = skeleton_y[sorted_indices]
        
        # Get middle 50% (25% to 75%)
        n_points = len(sorted_x)
        start_idx = int(n_points * 0.25)
        end_idx = int(n_points * 0.75)
        
        if end_idx <= start_idx:
            # Not enough points, use all
            middle_x = sorted_x
            middle_y = sorted_y
        else:
            middle_x = sorted_x[start_idx:end_idx]
            middle_y = sorted_y[start_idx:end_idx]
        
        # Get distances at middle skeleton points
        distances = mask_dist[middle_y, middle_x]
    else:
        # Get distances at all skeleton points
        distances = mask_dist[skeleton_y, skeleton_x]
    
    # Calculate thickness statistics (multiply by 2 to get full thickness)
    thickness_values = distances * 2
    
    thickness_stats = {
        'mean': np.mean(thickness_values),
        'median': np.median(thickness_values),
        'max': np.max(thickness_values),
        'std': np.std(thickness_values)
    }
    
    return thickness_stats, mask_dist


def calculate_cross_sectional_area(mask, cropped_dimensions=None):
    """
    Calculate cross-sectional area from the original extracted mask.
    
    For pixel measurement: counts the number of pixels in the mask.
    For mm measurement: resizes the mask from resized size (e.g., 224x224) 
    back to original cropped size. In the resized image, 10 mm = 48 pixels,
    so pixel_size = 10/48 = 0.2083 mm/pixel.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Binary segmentation mask (at resized size, e.g., 224x224)
    cropped_dimensions : tuple, optional
        Dimensions of the cropped image before resizing (height, width).
        Required for mm calculation.
    
    Returns:
    --------
    area_pixels : float
        Cross-sectional area in pixels (number of pixels in mask)
    area_mm : float, optional
        Cross-sectional area in mm² (if cropped_dimensions provided)
    """
    # Pixel area: simply count pixels
    area_pixels = float(np.sum(mask > 0.5))
    
    # MM area: resize mask to original cropped size, then calculate
    # In the resized image: 10 mm = 48 pixels
    area_mm = None
    if cropped_dimensions is not None:
        # Resize mask from current size to original cropped size
        from PIL import Image
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        resized_mask = mask_pil.resize(
            (cropped_dimensions[1], cropped_dimensions[0]),  # (width, height)
            Image.NEAREST  # Use nearest neighbor to preserve binary nature
        )
        resized_mask_np = (np.array(resized_mask) > 127).astype(np.uint8)
        
        # Count pixels in resized mask
        num_pixels = np.sum(resized_mask_np > 0.5)
        
        # Calculate area in mm²
        # In resized image: 10 mm = 48 pixels, so pixel_size = 10/48 mm/pixel
        pixel_size_mm = 10.0 / 48.0  # mm per pixel
        pixel_area_mm2 = pixel_size_mm * pixel_size_mm  # mm² per pixel
        area_mm = float(num_pixels * pixel_area_mm2)
    
    return area_pixels, area_mm


from scipy.interpolate import CubicSpline
import numpy as np
import torch.nn.functional as F
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter
import torch
from scipy.spatial import cKDTree
import pandas as pd
import json
import cv2

def scale_annotations(annotations, original_height = 487, original_width = 883, target_height = 128, target_width = 128):
    """
    Scales the coordinates of annotations based on the specified target dimensions.

    Args:
    - annotations (list of list of lists): Original annotations where each list contains
      sublists with [x, y] coordinates.
    - original_height (int): Original height of the images.
    - original_width (int): Original width of the images.
    - target_height (int): Target height to scale annotations to.
    - target_width (int): Target width to scale annotations to.

    Returns:
    - list: Scaled annotations in the same format as the input, with coordinates adjusted
      for the target dimensions.
    """
    height_scale = target_height / original_height
    width_scale = target_width / original_width
    scaled_annotations = []

    for annot in annotations:
        scaled_annotation = []
        for x, y in annot:
            scaled_x = int(x * width_scale)
            scaled_y = int(target_height - (y * height_scale))  # Adjust for non-Cartesian coordinate system
            scaled_annotation.append([scaled_x, scaled_y])
        scaled_annotations.append(scaled_annotation)

    return scaled_annotations

def remove_consecutive_duplicates(points):
    """
    Remove consecutive points that are identical from the list.
    
    Parameters:
    - points (list of list of int): List of [x, y] coordinate pairs
    
    Returns:
    - list of list of int: List with consecutive duplicates removed
    """
    if not points:
        return points
    unique_points = [points[0]]
    for point in points[1:]:
        if point != unique_points[-1]:
            unique_points.append(point)
    return unique_points

def parametric_interpolation(points, resolution=200):
    """
    Perform parametric interpolation to generate a smooth curve from a set of points.

    This function takes a list of coordinate pairs and an optional resolution parameter.
    It uses these points to generate a smooth curve by interpolating additional points between them.

    Parameters:
    - points (list of list of int): A list of [x, y] coordinate pairs. 
      Example: [[91, 60], [98, 71], ...]
    - resolution (int, optional): The number of points to interpolate between the given points
      to smooth the curve. Default is 1000.

    Returns:
    - list of list of int: A list of interpolated [x, y] coordinate pairs forming the smooth curve.

    Example:
    >>> parametric_interpolation([[91, 60], [98, 71]], resolution=500)
    [[91, 60], [91, 61], ..., [98, 71]]
    """

    points = remove_consecutive_duplicates(points)  

    if len(points) == 0:
        return np.array([]).reshape(0, 2) 
    
    # Convert the points to an array for easier manipulation
    points = np.array(points)
    
    # Calculate the cumulative distance (arc length) along the curve
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    
    t = distances / distances[-1]
    
    x = points[:, 0]
    y = points[:, 1]
    
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    
    # Create a fine parameterization for the output curve
    t_fine = np.linspace(0, 1, resolution)
    
    # Interpolate the x and y values along the curve
    x_interpolated = cs_x(t_fine)
    y_interpolated = cs_y(t_fine)
    
    # Return the interpolated curve
    return np.column_stack((x_interpolated, y_interpolated))



def generate_heatmap(height, width, points_x, points_y, sigma=3):
    """
    Generate a heatmap from a list of x and y coordinates.

    This function creates a heatmap with specified dimensions, places initial points on it, 
    then applies a Gaussian filter to spread the point intensity over a radius defined by sigma.

    Parameters:
    - height (int): The height of the heatmap.
    - width (int): The width of the heatmap.
    - points_x (list of int): List of x coordinates.
    - points_y (list of int): List of y coordinates.
    - sigma (int, optional): Standard deviation for Gaussian kernel. Default is 10.

    Returns:
    - numpy.ndarray: A 2D numpy array representing the generated heatmap, normalized to range [0, 1].
    """
    heatmap = np.zeros((height, width))
    for x, y in zip(points_x, points_y):
        heatmap[y, x] = 1
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    # Normalize the heatmap to range from 0 to 1
    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap /= max_value
    
    return heatmap

import numpy as np

def draw_circle(heatmap, center_x, center_y, radius, value=1, use_max=False):
    """
    Draw a filled circle on the heatmap array.

    Uses Euclidean distance (not skimage.draw.disk, which can rasterize small radii as squares).

    Parameters:
    - heatmap (numpy.ndarray): The heatmap on which to draw.
    - center_x (int): The x-coordinate of the circle's center.
    - center_y (int): The y-coordinate of the circle's center.
    - radius (int): The radius of the circle.
    - value (float): The value to set within the circle. Default is 1.
    - use_max (bool): If True, use pixel-wise maximum when overlapping stamps (for soft/hard mixes).
    """
    cx = int(np.asarray(center_x).item())
    cy = int(np.asarray(center_y).item())
    r = int(round(float(np.asarray(radius).item())))
    if r < 0:
        return
    val = float(value)
    y0 = int(max(0, cy - r))
    y1 = int(min(heatmap.shape[0], cy + r + 1))
    x0 = int(max(0, cx - r))
    x1 = int(min(heatmap.shape[1], cx + r + 1))
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                if use_max:
                    heatmap[y, x] = max(float(heatmap[y, x]), val)
                else:
                    heatmap[y, x] = val

def generate_heatmap_with_circles(height, width, points_x, points_y, radius=10):
    """
    Generate a heatmap from a list of x and y coordinates by drawing full circles.

    This function creates a heatmap with specified dimensions, places circles at initial points,
    then fills each circle to spread the point intensity over a radius defined by the radius parameter.

    Parameters:
    - height (int): The height of the heatmap.
    - width (int): The width of the heatmap.
    - points_x (list of int): List of x coordinates.
    - points_y (list of int): List of y coordinates.
    - radius (int, optional): Radius of the circles. Default is 10.

    Returns:
    - numpy.ndarray: A 2D numpy array representing the generated heatmap, normalized to range [0, 1].
    """
    heatmap = np.zeros((height, width))
    for x, y in zip(points_x, points_y):
        draw_circle(heatmap, x, y, radius)

    # Normalize the heatmap to range from 0 to 1
    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap /= max_value
    
    return heatmap



def perform_skeletonization(heatmap, threshold = 0.5):
    """
    Convert a heatmap to a binary skeletonized image.

    This function thresholds the heatmap to create a binary image and then applies
    skeletonization to reduce all objects to lines.

    Parameters:
    - heatmap (numpy.ndarray): Normalized heatmap as a 2D numpy array.

    Returns:
    - numpy.ndarray: A binary image where the structure has been reduced to lines.
    """
    binary_heatmap = heatmap > threshold  # 0.5 by default Assuming heatmap is normalized [0, 1]

    # Perform skeletonization
    skeleton = skeletonize(binary_heatmap)

    return skeleton


def annotations_to_heatmap(annot, height=128, width=128, resolution=150, type = 'gauss', param = 3):
    # Extract x and y coordinates from the given data
    if resolution != None: interpol_coords = parametric_interpolation(annot, resolution)
    else: interpol_coords = np.array(annot)

    x = [coord[0] for coord in interpol_coords]
    y = [coord[1] for coord in interpol_coords]

    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    if type == "gauss": heatmap = generate_heatmap(height, width, x, y, sigma= param)
    elif type == "circle": heatmap = generate_heatmap_with_circles (height, width, x, y, radius = param)

    return heatmap


def msd(sequence_u, sequence_v):
    """
    Calculates the Mean Sum of Distance (MSD) between two sequences of 2D points
    using the provided bidirectional formula.
    
    Parameters:
    sequence_u (np.array): A numpy array of 2D points representing sequence U.
    sequence_v (np.array): A numpy array of 2D points representing sequence V.
    
    Returns:
    float: The Mean Sum Distance between sequence U and sequence V.
    
    Each point in the sequences is expected to be a 2D point [x, y].
    """
    # Create KD-Trees for both sequences
    tree_u = cKDTree(sequence_u)
    tree_v = cKDTree(sequence_v)
    
    # Find nearest neighbor distances from U to V
    distances_u_to_v, _ = tree_v.query(sequence_u)
    
    # Find nearest neighbor distances from V to U
    distances_v_to_u, _ = tree_u.query(sequence_v)
    
    # Compute the mean of these distances
    sum_distance_u_to_v = np.sum(distances_u_to_v)
    sum_distance_v_to_u = np.sum(distances_v_to_u)
    
    # Calculate the final MSD value according to the given formula
    n = len(sequence_u)
    if n!= 0: msd_value = (1 / (2 * n)) * (sum_distance_u_to_v + sum_distance_v_to_u)
    else: msd_value = np.inf
    
    return msd_value

def skeleton_to_coordinates(heatmap):
    """
    Convert a skeletonized heatmap to a sequence of coordinates.
    
    Parameters:
    - heatmap: A numpy array with shape (240, 240) where 0s represent background and 1s represent the skeleton.
    
    Returns:
    - coordinates: A numpy array of coordinates where each row is a point (x, y).
    """
    # Extract the y, x positions of the 1s
    y_coords, x_coords = np.where(heatmap == 1)
    
    # Stack them into a two-column array: one for x, one for y
    coordinates = np.column_stack((x_coords, y_coords))
    
    return coordinates


def mini_batch_split(input_tensor, slice_size):
    """
    Splits a given input tensor into mini-batches of a specified size.

    This function divides an input tensor along its first dimension into smaller
    tensors (mini-batches) each containing 'slice_size' elements. If the total
    number of elements in the input tensor isn't a multiple of 'slice_size',
    the last mini-batch will contain the remaining elements, even if it's less
    than 'slice_size'.

    Parameters:
    input_tensor (torch.Tensor): The tensor to be split into mini-batches.
    slice_size (int): The size of each mini-batch.

    Returns:
    torch.Tensor: A tensor containing the mini-batches as its elements.
    Each mini-batch is a sub-tensor of the input tensor with 'slice_size'
    elements along the first dimension, except potentially the last one which
    might contain fewer elements if the total size is not a multiple of
    'slice_size'.

    Example:
    >>> input_tensor = torch.arange(24).reshape(6, 4)
    >>> mini_batch_split(input_tensor, 2)
    tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7]],

            [[ 8,  9, 10, 11],
             [12, 13, 14, 15]],

            [[16, 17, 18, 19],
             [20, 21, 22, 23]]])
    """
    temp_list = []
    for i in range(input_tensor.shape[0] // slice_size):
        start_idx = i * slice_size
        end_idx = start_idx + slice_size
        temp_list.append(input_tensor[start_idx:end_idx])

    # Take the last slice_size elements if the total number is not divisible by slice_size
    if input_tensor.shape[0] % slice_size != 0:
        start_idx = input_tensor.shape[0] - slice_size
        temp_list.append(input_tensor[start_idx:])

    out_tensor = torch.stack(temp_list)  # Use torch.stack to maintain the correct tensor structure
    return out_tensor

def mini_batch_split_list(input_list, slice_size):
    temp_list = []
    total_elements = len(input_list)
    
    for i in range(total_elements // slice_size):
        start_idx = i * slice_size
        end_idx = start_idx + slice_size
        temp_list.append(input_list[start_idx:end_idx])

    # Take the last slice_size elements if the total number is not divisible by slice_size
    if total_elements % slice_size != 0:
        start_idx = total_elements - slice_size
        temp_list.append(input_list[start_idx:])

    return temp_list

def gaussian_kernel(size, sigma):
    """
    Generates a 1D Gaussian kernel.
    
    Parameters:
    size (int): The size of the kernel.
    sigma (float): The standard deviation of the Gaussian distribution.
    
    Returns:
    torch.Tensor: The Gaussian kernel of shape [size].
    """
    # Create a 1D tensor with values centered around zero
    x = torch.arange(size) - (size - 1) / 2.0
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()  # Normalize the kernel
    return kernel

def apply_gaussian_filter(tensor, sigma=3):
    """
    Applies a Gaussian filter along the first axis of a tensor.
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape [N, H, W]
    sigma (float): The standard deviation of the Gaussian distribution.
    
    Returns:
    torch.Tensor: The tensor after applying the Gaussian filter.
    """
    N, H, W = tensor.shape
    kernel = gaussian_kernel(N, sigma).to(tensor.device)
    
    # Reshape kernel for broadcasting
    kernel = kernel.view(N, 1, 1)
    
    # Apply the Gaussian filter along the first axis
    filtered_tensor = tensor * kernel

    return filtered_tensor

def normalize_slices(tensor):
    """
    Normalize each 2D slice of a 3D tensor to the range [0, 1].
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape [N, H, W]
    
    Returns:
    torch.Tensor: Normalized tensor of shape [N, H, W]
    """
    N, H, W = tensor.shape
    
    # Initialize the normalized tensor with the same shape
    normalized_tensor = torch.empty_like(tensor)
    
    for i in range(N):
        # Get the slice
        slice_2d = tensor[i]
        
        # Normalize the slice
        min_val = slice_2d.min()
        max_val = slice_2d.max()
        normalized_tensor[i] = (slice_2d - min_val) / (max_val - min_val)
    
    return normalized_tensor


def gaussian_sliding_window(tensor, window_size, step=1, sigma=3):
    """
    Create a sliding window view of the input tensor, normalizes each window and then applies Gaussian filter. 
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape [N, H, W]
    window_size (int): Size of the sliding window along the first dimension (N)
    step (int): Step size for the sliding window. Default is 1.
    
    Returns:
    torch.Tensor: A tensor of shape [num_windows, window_size, H, W]
    """
    N, H, W = tensor.shape

    # Compute the number of windows
    num_windows = (N - window_size) // step + 1

    # Initialize the output tensor
    windows = torch.empty((num_windows, window_size, H, W), dtype=tensor.dtype, device=tensor.device)

    # Populate the output tensor with sliding windows
    for i in range(num_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        window = normalize_slices(tensor[start_idx:end_idx]) # Normalize from 0 to 1
        windows[i] = apply_gaussian_filter(window, sigma)
    return windows


def sliding_window_split(tensor, window_size, step=1):
    """
    Create a sliding window view of the input tensor.
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape [N, H, W]
    window_size (int): Size of the sliding window along the first dimension (N)
    step (int): Step size for the sliding window. Default is 1.
    
    Returns:
    torch.Tensor: A tensor of shape [num_windows, window_size, H, W]
    """
    N, H, W = tensor.shape

    # Compute the number of windows
    num_windows = (N - window_size) // step + 1

    # Initialize the output tensor
    windows = torch.empty((num_windows, window_size, H, W), dtype=tensor.dtype, device=tensor.device)

    # Populate the output tensor with sliding windows
    for i in range(num_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        windows[i] = tensor[start_idx:end_idx]

    return windows

def heatmaps_for_sliding_window(tensor, window_size, step=1):
    """
    Create a sliding window view of the input tensor and select the central 2D tensor from each window.
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape [N, H, W]
    window_size (int): Size of the sliding window along the first dimension (N)
    step (int): Step size for the sliding window. Default is 1.
    
    Returns:
    torch.Tensor: A tensor of shape [num_windows, 1, H, W], containing the central 2D tensor from each window
    """
    N, H, W = tensor.shape

    # Compute the number of windows
    num_windows = (N - window_size) // step + 1
    
    # Compute the central index of the window
    central_index = window_size // 2

    # Initialize the output tensor
    central_slices = torch.empty((num_windows, 1, H, W), dtype=tensor.dtype, device=tensor.device)

    # Populate the output tensor with the central slices from each window
    for i in range(num_windows):
        start_idx = i * step
        central_slices[i] = tensor[start_idx + central_index].unsqueeze(0)

    return central_slices

def resize_coordinates_list(contours, original_size, target_size):
    """
    Resize coordinates proportionally from original size to target size.
    
    Parameters:
    contours (list of lists): List of contour coordinates, where each contour is a list of (x, y) tuples.
    original_size (tuple): Original size of the image as (width, height).
    target_size (tuple): Target size of the image as (width, height).
    
    Returns:
    list of lists: Resized contour coordinates maintaining the same format.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    resized_contours = []
    for contour in contours:
        resized_contour = [(int(x * target_w / orig_w), int(y * target_h / orig_h)) for x, y in contour]
        resized_contours.append(resized_contour)
    
    return resized_contours

def resize_contour(contour, original_size, target_size):
    """
    Resize coordinates proportionally from original size to target size.
    
    Parameters:
    contour (list of tuples): List of (x, y) coordinate tuples for a single picture.
    original_size (tuple): Original size of the image as (width, height).
    target_size (tuple): Target size of the image as (width, height).
    
    Returns:
    list of tuples: Resized coordinates maintaining the same format.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    resized_contour = [[int(x * target_w / orig_w), int(y * target_h / orig_h)] for x, y in contour]
    
    return resized_contour

def resize_coordinates(input_json_path, output_json_path, original_size, new_size):
    """
    Resize coordinates from the original images size to a new size.

    Parameters:
    - input_json_path (str): Path to the input JSON file containing the original coordinates.
    - output_json_path (str): Path to save the JSON file with resized coordinates.
    - original_size (tuple): The original size as a (width, height) tuple.
    - new_size (tuple): The new size as a (width, height) tuple.
    """
    # Load the original coordinates from the JSON file
    with open(input_json_path, 'r') as file:
        data = json.load(file)
    
    # Calculate the scaling factors
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    
    # Resize the coordinates
    resized_data = []
    for image_coords in data:
        resized_image_coords = []
        for point in image_coords:
            resized_point = [point[0] * scale_x, point[1] * scale_y]
            resized_image_coords.append(resized_point)
        resized_data.append(resized_image_coords)
    
    # Save the resized coordinates to a new JSON file
    with open(output_json_path, 'w') as file:
        json.dump(resized_data, file)


def resize_images(images, target_shape, save_file=None):
    """
    Resize a numpy array of images to a given shape.

    Parameters:
    - images (numpy array): The input array of images with shape [N, 512, 512].
    - target_shape (tuple): The target shape (height, width) for each image.
    - save_file (str, optional): The filename to save the resized images as a .npy file.
    
    Returns:
    - resized_images (numpy array): The resized array of images with shape [N, target_shape[0], target_shape[1]].
    """
    N = images.shape[0]
    resized_images = np.zeros((N, target_shape[0], target_shape[1]), dtype=images.dtype)

    for i in range(N):
        resized_images[i] = cv2.resize(images[i], (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

    if save_file:
        np.save(save_file, resized_images)

    return resized_images


import numpy as np

def histogram_match_tensor_array(images, reference_cdf, ignore_zero=True):
    """
    Matches histograms of all images in a PyTorch tensor array to the reference CDF.
    
    Args:
        images (torch.Tensor): PyTorch tensor of shape (N, H, W), where N is the number of images.
        reference_cdf (np.ndarray): Reference CDF from the saved histogram data.
        ignore_zero (bool): If True, ignore zero values in the histogram matching. Default is True.
    
    Returns:
        torch.Tensor: Tensor of histogram-matched images with the same shape as the input.
    """
    import torch
    
    num_bins = 256  # For 8-bit grayscale images
    matched_images = torch.zeros_like(images, dtype=torch.float32)  # Create an output tensor
    
    # Convert the input tensor to NumPy for processing
    images_np = images.numpy() if isinstance(images, torch.Tensor) else images

    for i, image in enumerate(images_np):
        # Ensure the image is in the correct format (NumPy array, with integer pixel values)
        if not np.issubdtype(image.dtype, np.integer):
            image = (image * 255).astype(np.uint8)  # Scale [0, 1] tensors to [0, 255] integers
        
        # Compute the histogram and CDF of the target image
        target_histogram, _ = np.histogram(image.ravel(), bins=num_bins, range=(0, 256))
        target_cdf = np.cumsum(target_histogram)
        target_cdf = target_cdf / target_cdf[-1]  # Normalize to [0, 1]
        
        # Create a mapping from target intensities to reference intensities
        mapping = np.zeros(num_bins, dtype=np.uint8)
        for target_intensity in range(num_bins):
            if ignore_zero and target_intensity == 0:
                mapping[target_intensity] = 0
            else:
                # Find the closest match in the reference CDF
                diff = np.abs(reference_cdf - target_cdf[target_intensity])
                closest_match = np.argmin(diff)
                mapping[target_intensity] = closest_match
        
        # Apply the mapping to the image
        matched_image = mapping[image]
        
        # Store the result back in the output tensor
        matched_images[i] = torch.from_numpy(matched_image.astype(np.float32)) / 255.0  # Scale back to [0, 1]
    
    return matched_images


def match_histogram_single(image, reference_cdf):
    """
    Matches the histogram of a single image to the reference CDF, ignoring zero values.

    Args:
        image (np.ndarray): Input grayscale image to be matched.
        reference_cdf (np.ndarray): Reference CDF.

    Returns:
        np.ndarray: The histogram-matched image.
    """
    # Compute the histogram and CDF of the target image, ignoring zero values
    num_bins = 256
    mask = image > 0
    target_histogram, _ = np.histogram(image[mask].ravel(), bins=num_bins, range=(0, 256))
    target_cdf = np.cumsum(target_histogram)
    target_cdf = target_cdf / target_cdf[-1]  # Normalize to [0, 1]

    # Create a mapping from target intensities to reference intensities
    mapping = np.zeros(num_bins, dtype=np.uint8)
    for target_intensity in range(1, num_bins):  # Start from 1 to ignore zero values
        # Find the closest match in the reference CDF
        diff = np.abs(reference_cdf - target_cdf[target_intensity])
        closest_match = np.argmin(diff)
        mapping[target_intensity] = closest_match

    # Apply the mapping to generate the matched image
    matched_image = image.copy()
    # Convert pixel values to integers for indexing (clip to valid range)
    image_indices = np.clip(image[mask].astype(np.int32), 0, 255)
    matched_image[mask] = mapping[image_indices]
    return matched_image


def annotations_to_heatmap_circle(annot, height=128, width=128, resolution=150, type = 'gauss', param = 3):
    # Extract x and y coordinates from the given data
    if resolution != None: interpol_coords = parametric_interpolation(annot, resolution)
    else: interpol_coords = np.array(annot)

    x = [coord[0] for coord in interpol_coords]
    y = [coord[1] for coord in interpol_coords]

    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    if type == "gauss": heatmap = generate_heatmap(height, width, x, y, sigma= param)
    elif type == "circle": heatmap = generate_heatmap_with_circles (height, width, x, y, radius = param)

    return heatmap

def resize_contour(contour, original_size, target_size):
    """
    Resize coordinates proportionally from original size to target size.
    
    Parameters:
    contour (list of tuples): List of (x, y) coordinate tuples for a single picture.
    original_size (tuple): Original size of the image as (width, height).
    target_size (tuple): Target size of the image as (width, height).
    
    Returns:
    list of tuples: Resized coordinates maintaining the same format.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    resized_contour = [[int(x * target_w / orig_w), int(y * target_h / orig_h)] for x, y in contour]
    
    return resized_contour
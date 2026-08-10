import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from pathlib import Path
import torchvision.transforms as transforms
from scipy.ndimage import label
from utils.processing import resize_contour, annotations_to_heatmap

def _zoom_image(crop_percentages):
    """
    Zoom the image by a factor of 2 by adjusting crop percentages.
    
    This function applies the same zoom transformation as in dataset.py.
    When zoom is 2x, the cropped region becomes smaller (more is cropped from edges).
    
    Parameters:
    -----------
    crop_percentages : tuple
        Original crop percentages (top, bottom, left, right)
    
    Returns:
    --------
    zoomed_crop_percentages : tuple
        Zoomed crop percentages (top, bottom, left, right)
    """
    top_pct, bottom_pct, left_pct, right_pct = crop_percentages
    zoom_vert = (1 - top_pct - bottom_pct) / 2
    zoom_horz = (1 - left_pct - right_pct) / 2
    return top_pct + zoom_vert, bottom_pct, left_pct + zoom_horz/2, right_pct + zoom_horz/2


class MuscleSegmentationDataset(Dataset):
    """
    Dataset for muscle segmentation from PNG images and masks.
    
    Directory structure expected:
    root_dir/
        images/
            image1.png
            image2.png
            ...
        masks/
            image1.png
            image2.png
            ...
    """
    
    def __init__(self, root_dir, transform=None, image_size=None, crop_percentages=(0.26, 0.065, 0.175, 0.17), 
                 augmented_samples=None, smooth_mask=True, zoom=False, image_transform=None,
                 mask_transform=None):
        """
        Parameters:
        -----------
        root_dir : str
            Root directory containing 'images' and 'masks' folders
        transform : callable, optional
            Optional transform to be applied on BOTH image and mask
        image_size : tuple, optional
            Target size (height, width) to resize images and masks. If None, uses original size.
        crop_percentages : tuple, default=(0.26, 0.065, 0.175, 0.17),
            Crop percentages for (top, bottom, left, right) respectively
        augmented_samples : list, optional
            List of offline augmented samples (stored in memory)
        smooth_mask : bool, default=True
            If True, smooth the mask edges using the smart method
        zoom : bool, default=False
            If True, apply 2x zoom transformation to crop_percentages
        image_transform : callable, optional
            Optional transform to be applied ONLY on image
        mask_transform : callable, optional
            Optional transform to be applied ONLY on mask
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.crop_percentages = crop_percentages  # (top, bottom, left, right)
        self.augmented_samples = augmented_samples if augmented_samples is not None else []
        self.smooth_mask = smooth_mask
        self.zoom = zoom
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        # Find all image-mask pairs
        self.samples = self._find_image_mask_pairs()

        if self.zoom:
            self.crop_percentages = _zoom_image(self.crop_percentages)
            print(f"Zoomed crop percentages: {self.crop_percentages}")

        if len(self.samples) == 0 and len(self.augmented_samples) == 0:
            raise ValueError(f"No image-mask pairs found in {root_dir}")
        
        print(f"Found {len(self.samples)} original images")
        if len(self.augmented_samples) > 0:
            print(f"Found {len(self.augmented_samples)} augmented samples")

    def _find_image_mask_pairs(self):
        """Scan directory structure and find all (image, mask) PNG pairs."""
        samples = []
        
        images_dir = self.root_dir / 'images'
        masks_dir = self.root_dir / 'masks'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise ValueError(f"Masks directory not found: {masks_dir}")
        
        skip_prefixes = ('._', '.DS_Store', 'Thumbs.db', 'desktop.ini')
        
        # Find all PNG images
        for img_path in images_dir.glob("*.png"):
            if any(img_path.name.startswith(prefix) for prefix in skip_prefixes):
                continue
            
            if img_path.name.startswith('.'):
                continue
            
            # Check if corresponding mask exists
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                samples.append({
                    'image_path': str(img_path),
                    'mask_path': str(mask_path)
                })
        
        return samples
    
    def _load_json_annotation(self, json_path):
        """Load JSON annotation and extract polygon points."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _polygon_to_mask(self, points, img_height, img_width):
        """
        Convert polygon points to binary mask using PIL.
        
        Parameters:
        -----------
        points : list of [x, y] coordinates
        img_height : int
        img_width : int
        
        Returns:
        --------
        mask : np.ndarray, binary mask
        """
        # Create a PIL image and draw the polygon
        mask_img = Image.new('L', (img_width, img_height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # Convert points to tuple format for PIL
        polygon_points = [(p[0], p[1]) for p in points]
        draw.polygon(polygon_points, outline=1, fill=1)
        
        # Convert to numpy array
        mask = np.array(mask_img, dtype=np.uint8)
        
        return mask
    
    def _crop_and_adjust_points(self, points, original_width, original_height, crop_percentages):
        """
        Crop points based on crop percentages and adjust coordinates.
        
        Parameters:
        -----------
        points : list of [x, y] coordinates
        original_width : int
        original_height : int
        crop_percentages : tuple
            (top, bottom, left, right) percentages
        
        Returns:
        --------
        cropped_points : list of [x, y] coordinates in cropped image space
        crop_box : tuple
            (left, top, right, bottom) in original image coordinates
        """
        top_pct, bottom_pct, left_pct, right_pct = crop_percentages
        
        # Calculate crop box in original image coordinates
        left = int(original_width * left_pct)
        right = int(original_width * (1 - right_pct))
        top = int(original_height * top_pct)
        bottom = int(original_height * (1 - bottom_pct))
        
        crop_box = (left, top, right, bottom)
        
        # Adjust points to cropped image space
        cropped_points = []
        for point in points:
            x, y = point[0], point[1]
            # Translate coordinates relative to crop box
            new_x = x - left
            new_y = y - top
            cropped_points.append([new_x, new_y])
        
        return cropped_points, crop_box
    
    def _create_segmentation_mask(self, json_data, img_height, img_width, crop_percentages=None):
        """
        Create binary segmentation mask from JSON annotations.
        Combines all polygon shapes into a single mask.
        
        Parameters:
        -----------
        json_data : dict
            Loaded JSON annotation data
        img_height : int
        img_width : int
        crop_percentages : tuple, optional
            If provided, adjusts points for cropping
        
        Returns:
        --------
        mask : np.ndarray, binary mask
        """
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Get original image dimensions from JSON if available
        original_height = json_data.get('imageHeight', img_height)
        original_width = json_data.get('imageWidth', img_width)
        
        # Combine all shapes into single mask
        for shape in json_data.get('shapes', []):
            if shape.get('shape_type') == 'polygon':
                points = shape['points']
                
                # Adjust points if cropping is needed
                if crop_percentages is not None:
                    points, _ = self._crop_and_adjust_points(
                        points, original_width, original_height, crop_percentages
                    )
                
                shape_mask = self._polygon_to_mask(points, img_height, img_width)
                mask = np.maximum(mask, shape_mask)  # Combine masks

        if self.smooth_mask:
            # Lazy import to avoid circular dependency
            from methods.mask import smooth_mask_edges
            mask = smooth_mask_edges(mask)
        return mask
    
    def _get_largest_connected_component(self, image):
        """
        Extract the largest connected component from an image, where 0 is background.
        Removes smaller components separated by background.
        
        Parameters:
        -----------
        image : np.ndarray
            Image array (can be any dtype, 0 = background)
        
        Returns:
        --------
        processed_image : np.ndarray
            Image with only the largest connected component, same dtype as input
        """
        # Create binary mask: non-zero pixels are foreground
        binary_mask = (image != 0).astype(np.uint8)
        
        # Label connected components
        labeled_mask, num_features = label(binary_mask)
        
        if num_features == 0:
            # No non-zero pixels, return zeros
            return np.zeros_like(image)
        
        # Find the largest component
        component_sizes = []
        for i in range(1, num_features + 1):
            component_sizes.append((labeled_mask == i).sum())
        
        largest_component_id = np.argmax(component_sizes) + 1
        largest_component_mask = (labeled_mask == largest_component_id)
        
        # Apply mask to original image: keep only largest component, set rest to 0
        processed_image = image.copy()
        processed_image[~largest_component_mask] = 0
        
        return processed_image
    
    def __len__(self):
        return len(self.samples) + len(self.augmented_samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Check if this is an augmented sample
        if idx >= len(self.samples):
            # Return augmented sample from memory
            aug_idx = idx - len(self.samples)
            aug_sample = self.augmented_samples[aug_idx]
            
            image = aug_sample['image'].copy()
            mask = aug_sample['mask'].copy()
            
            # Preprocess image: keep only largest connected component (0 = background)
            # Convert to uint8 if needed for processing, then back
            if image.dtype == np.float32:
                # Temporarily convert to uint8 for processing
                image_uint8 = (image * 255.0).astype(np.uint8)
                image_uint8 = self._get_largest_connected_component(image_uint8)
                image = image_uint8.astype(np.float32) / 255.0
            else:
                image = self._get_largest_connected_component(image)
            
            # Ensure mask is binary
            mask = (mask > 0.5).astype(np.float32)
            
            # Convert to torch tensors
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)

            # Separate transforms
            if self.image_transform: image = self.image_transform(image)
            if self.mask_transform: mask = self.mask_transform(mask)
            
            # Together transforms
            if self.transform:
                stacked = torch.cat([image, mask], dim=0)
                stacked = self.transform(stacked)
                image = stacked[0:1]
                mask = stacked[1:2]
            
            return image, mask
        
        # Regular sample - load from disk
        img_path = self.samples[idx]['image_path']
        mask_path = self.samples[idx]['mask_path']
        
        # Try to open image
        try:
            image = Image.open(img_path)
            # Load the image to verify it's valid (this forces PIL to read the file)
            image.load()
        except Exception as e:
            # If image loading fails, raise error with helpful message
            raise ValueError(f"Failed to load image {img_path}: {str(e)}. "
                           f"This might be a corrupted file or system metadata file.")
        
        # Convert to grayscale (for ultrasound data)
        if image.mode != 'L':
            image = image.convert('L')
        
        # Load mask
        try:
            mask = Image.open(mask_path)
            mask.load()
        except Exception as e:
            raise ValueError(f"Failed to load mask {mask_path}: {str(e)}. "
                           f"This might be a corrupted file or system metadata file.")
        
        # Convert mask to grayscale if needed
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Get original image dimensions
        original_width, original_height = image.size
        
        # Crop image
        top_pct, bottom_pct, left_pct, right_pct = self.crop_percentages
        left = int(original_width * left_pct)
        right = int(original_width * (1 - right_pct))
        top = int(original_height * top_pct)
        bottom = int(original_height * (1 - bottom_pct))
        
        # Crop image and mask
        image = image.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))
        
        # Resize if specified
        if self.image_size is not None:
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        
        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        # Preprocess image: keep only largest connected component (0 = background)
        image = self._get_largest_connected_component(image)
        
        # Normalize image to [0, 1] and convert to float
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Convert mask from 0/255 to binary 0/1 (one-hot encoding)
        # Masks are stored as 0 or 255, convert to 0 or 1
        mask = (mask > 127).astype(np.float32)  # Threshold at 127 to handle 0/255
        
        # Convert to torch tensors
        # Image: (H, W) -> (1, H, W) for grayscale
        image = torch.from_numpy(image).unsqueeze(0)
        
        # Mask: (H, W) -> (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Separate transforms (applied before joint transform)
        if self.image_transform: image = self.image_transform(image)
        if self.mask_transform: mask = self.mask_transform(mask)
        
        # Apply transforms if provided (online augmentations)
        if self.transform:
            # For custom transforms that handle image and mask together
            if hasattr(self.transform, '__call__'):
                # Try to call as a function that takes (image, mask)
                try:
                    image, mask = self.transform(image, mask)
                except TypeError:
                    # Fallback to stacked tensor approach
                    stacked = torch.cat([image, mask], dim=0)
                    stacked = self.transform(stacked)
                    image = stacked[0:1]  # First channel is image
                    mask = stacked[1:2]  # Second channel is mask
        
        return image, mask


def dataset_zoom_check(dataset):
    """
    Check if all annotation masks are within the zoomed area.
    
    This function verifies that all polygon points from the JSON annotations
    fall within the cropped region when zoom is enabled. It reports any samples
    where masks extend outside the zoomed crop area.
    
    Parameters:
    -----------
    dataset : MuscleSegmentationDataset
        An instance of MuscleSegmentationDataset to check
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'all_within': bool, True if all masks are within zoomed area
        - 'total_samples': int, Total number of samples checked
        - 'samples_outside': list, List of sample indices and paths where masks extend outside
        - 'zoom_enabled': bool, Whether zoom was enabled in the dataset
    """
    if not isinstance(dataset, MuscleSegmentationDataset):
        raise TypeError("dataset must be an instance of MuscleSegmentationDataset")
    
    results = {
        'all_within': True,
        'total_samples': 0,
        'samples_outside': [],
        'zoom_enabled': dataset.zoom
    }
    
    # If zoom is not enabled, all masks should be fine (no zoom applied)
    if not dataset.zoom:
        print("Zoom is not enabled in the dataset. All masks should be within the crop area.")
        results['all_within'] = True
        return results
    
    # Get the zoomed crop percentages
    top_pct, bottom_pct, left_pct, right_pct = dataset.crop_percentages
    
    print(f"Checking {len(dataset.samples)} samples with zoomed crop percentages: "
          f"top={top_pct:.4f}, bottom={bottom_pct:.4f}, left={left_pct:.4f}, right={right_pct:.4f}")
    
    # Check each sample
    for idx, sample in enumerate(dataset.samples):
        results['total_samples'] += 1
        
        # Load the image to get original dimensions
        try:
            image = Image.open(sample['image_path'])
            original_width, original_height = image.size
        except Exception as e:
            print(f"Warning: Could not load image {sample['image_path']}: {e}")
            continue
        
        # Calculate the crop box boundaries in original image coordinates
        left_bound = int(original_width * left_pct)
        right_bound = int(original_width * (1 - right_pct))
        top_bound = int(original_height * top_pct)
        bottom_bound = int(original_height * (1 - bottom_pct))
        
        # Load JSON annotation
        try:
            json_data = dataset._load_json_annotation(sample['json_path'])
        except Exception as e:
            print(f"Warning: Could not load JSON {sample['json_path']}: {e}")
            continue
        
        # Check all polygon shapes
        mask_outside = False
        outside_points_info = []
        
        for shape_idx, shape in enumerate(json_data.get('shapes', [])):
            if shape.get('shape_type') == 'polygon':
                points = shape['points']
                
                # Check each point in the polygon
                for point_idx, point in enumerate(points):
                    x, y = point[0], point[1]
                    
                    # Check if point is outside the crop box
                    if (x < left_bound or x > right_bound or 
                        y < top_bound or y > bottom_bound):
                        mask_outside = True
                        outside_points_info.append({
                            'shape_idx': shape_idx,
                            'point_idx': point_idx,
                            'point': [x, y],
                            'bounds': {
                                'left': left_bound,
                                'right': right_bound,
                                'top': top_bound,
                                'bottom': bottom_bound
                            }
                        })
        
        if mask_outside:
            results['all_within'] = False
            results['samples_outside'].append({
                'index': idx,
                'image_path': sample['image_path'],
                'json_path': sample['json_path'],
                'original_size': (original_width, original_height),
                'crop_box': (left_bound, top_bound, right_bound, bottom_bound),
                'outside_points': outside_points_info
            })
    
    # Print summary
    if results['all_within']:
        print(f"\n✓ All {results['total_samples']} samples have masks within the zoomed area.")
    else:
        print(f"\n✗ Found {len(results['samples_outside'])} samples with masks extending outside the zoomed area:")
        for sample_info in results['samples_outside']:
            print(f"  - Sample {sample_info['index']}: {sample_info['image_path']}")
            print(f"    Original size: {sample_info['original_size']}")
            print(f"    Crop box: left={sample_info['crop_box'][0]}, top={sample_info['crop_box'][1]}, "
                  f"right={sample_info['crop_box'][2]}, bottom={sample_info['crop_box'][3]}")
            print(f"    {len(sample_info['outside_points'])} point(s) outside bounds")
    
    return results


def load_contour_data(image_folder, contour_folder, type="circle", height=128, width=128):
    """
    Loads ultrasound images and contour annotations, resizes both the images and contours to the specified dimensions,
    and converts both images and annotations into tensors.

    Args:
        image_folder (str): Path to the folder containing PNG ultrasound images.
        contour_folder (str): Path to the folder containing JSON contour annotations.
        height (int): Desired height for the resized images and heatmaps.
        width (int): Desired width for the resized images and heatmaps.

    Returns:
        (torch.Tensor, torch.Tensor, list, list):
        - images_tensor: Tensor of resized ultrasound images (shape: [N, H, W]).
        - heatmaps_tensor: Tensor of heatmaps generated from resized contour annotations (shape: [N, H, W]).
        - resized_contours: List of resized contour annotations.
        - stems: List of sample stems (PNG basename without extension), same length as tensors.
    """
    # Get sorted lists of image and contour files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    contour_files = sorted([f for f in os.listdir(contour_folder) if f.endswith('.json')])

    images = []
    heatmaps = []
    resized_contours = []
    stems = []

    for image_file, contour_file in zip(image_files, contour_files):
        if image_file.replace('.png', '') == contour_file.replace('.json', ''):
            stem = image_file.replace(".png", "")
            # Load image and get original size
            image_path = os.path.join(image_folder, image_file)
            with Image.open(image_path) as img:
                orig_size = img.size  # Original size as (width, height)
                img_resized = img.resize((width, height), Image.BILINEAR)  # Resize to (width, height)
                images.append(np.array(img_resized))  # Convert to NumPy array

            # Load contour and resize it
            contour_path = os.path.join(contour_folder, contour_file)
            with open(contour_path, 'r') as f:
                annot = json.load(f)
                if len(annot) > 0:  # Only resize if there are valid annotations
                    resized_contour = resize_contour(annot, orig_size, (width, height))
                else:
                    resized_contour = []  # Keep empty if no annotations
                resized_contours.append(resized_contour)

                # Generate heatmap from resized contour
                heatmap = annotations_to_heatmap(resized_contour, height=height, width=width, resolution=None, type=type)
                heatmaps.append(heatmap)
                stems.append(stem)

    # Convert images and heatmaps to PyTorch tensors
    images_tensor = torch.tensor(np.array(images), dtype=torch.float32) / 255.0
    heatmaps_tensor = torch.tensor(heatmaps, dtype=torch.float32)

    return images_tensor, heatmaps_tensor, resized_contours, stems
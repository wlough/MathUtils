# #!/usr/bin/env python3
# """
# Nuclear Envelope Z-Stack Processor

# This script processes TIFF z-stack files containing nuclear envelope during mitosis
# to perform:
# 1. Image segmentation (inside/outside nuclear envelope)
# 2. Boundary extraction from each z-slice
# 3. 3D point cloud construction from boundaries

# Author: Generated for MathUtils project
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import tifffile
# import cv2
# from scipy import ndimage
# from scipy.spatial import distance
# from skimage import filters, morphology, measure, segmentation
# from skimage.feature import canny
# from skimage.morphology import binary_erosion, binary_dilation, disk
# import warnings

# warnings.filterwarnings("ignore")


# class NuclearEnvelopeProcessor:
#     """
#     Process nuclear envelope z-stack TIFF files for segmentation and boundary extraction.
#     """

#     def __init__(
#         self,
#         gaussian_sigma=1.0,
#         min_area=100,
#         max_area=10000,
#         canny_sigma=1.0,
#         canny_low_threshold=0.1,
#         canny_high_threshold=0.2,
#         morphology_disk_size=3,
#     ):
#         """
#         Initialize processor with configurable parameters.

#         Parameters:
#         -----------
#         gaussian_sigma : float
#             Standard deviation for Gaussian blur preprocessing
#         min_area : int
#             Minimum area for connected components
#         max_area : int
#             Maximum area for connected components
#         canny_sigma : float
#             Standard deviation for Gaussian filter in Canny edge detection
#         canny_low_threshold : float
#             Lower threshold for Canny edge detection
#         canny_high_threshold : float
#             Upper threshold for Canny edge detection
#         morphology_disk_size : int
#             Size of disk structuring element for morphological operations
#         """
#         self.gaussian_sigma = gaussian_sigma
#         self.min_area = min_area
#         self.max_area = max_area
#         self.canny_sigma = canny_sigma
#         self.canny_low_threshold = canny_low_threshold
#         self.canny_high_threshold = canny_high_threshold
#         self.morphology_disk_size = morphology_disk_size

#     def load_tiff_stack(self, filepath):
#         """
#         Load TIFF z-stack file.

#         Parameters:
#         -----------
#         filepath : str or Path
#             Path to TIFF file

#         Returns:
#         --------
#         numpy.ndarray
#             3D array with shape (z, y, x)
#         """
#         filepath = Path(filepath)
#         if not filepath.exists():
#             raise FileNotFoundError(f"File not found: {filepath}")

#         try:
#             # Load the TIFF stack
#             stack = tifffile.imread(str(filepath))

#             # Ensure we have a 3D array (z, y, x)
#             if stack.ndim == 2:
#                 stack = stack[np.newaxis, :, :]
#             elif stack.ndim == 4:
#                 # If RGBA or similar, take first channel
#                 stack = stack[:, :, :, 0]

#             print(f"Loaded z-stack with shape: {stack.shape}")
#             return stack

#         except Exception as e:
#             raise ValueError(f"Error loading TIFF file: {e}")

#     def preprocess_slice(self, image_slice):
#         """
#         Preprocess a single z-slice.

#         Parameters:
#         -----------
#         image_slice : numpy.ndarray
#             2D image slice

#         Returns:
#         --------
#         numpy.ndarray
#             Preprocessed image slice
#         """
#         # Normalize to 0-1 range
#         if image_slice.max() > 1:
#             image_slice = image_slice.astype(np.float32) / image_slice.max()

#         # Apply Gaussian blur to reduce noise
#         smoothed = filters.gaussian(image_slice, sigma=self.gaussian_sigma)

#         return smoothed

#     def segment_nuclear_envelope(self, image_slice):
#         """
#         Segment nuclear envelope from a single z-slice using multiple methods.

#         Parameters:
#         -----------
#         image_slice : numpy.ndarray
#             2D preprocessed image slice

#         Returns:
#         --------
#         numpy.ndarray
#             Binary mask where True indicates inside nuclear envelope
#         """
#         # Method 1: Otsu thresholding
#         thresh_otsu = filters.threshold_otsu(image_slice)
#         binary_otsu = image_slice > thresh_otsu

#         # Method 2: Li thresholding (often good for fluorescence)
#         try:
#             thresh_li = filters.threshold_li(image_slice)
#             binary_li = image_slice > thresh_li
#         except:
#             binary_li = binary_otsu

#         # Method 3: Triangle thresholding
#         try:
#             thresh_triangle = filters.threshold_triangle(image_slice)
#             binary_triangle = image_slice > thresh_triangle
#         except:
#             binary_triangle = binary_otsu

#         # Combine methods by taking intersection
#         binary_combined = binary_otsu & binary_li & binary_triangle

#         # Clean up the binary image
#         binary_cleaned = self._clean_binary_image(binary_combined)

#         return binary_cleaned

#     def _clean_binary_image(self, binary_image):
#         """
#         Clean binary image using morphological operations.

#         Parameters:
#         -----------
#         binary_image : numpy.ndarray
#             Binary image to clean

#         Returns:
#         --------
#         numpy.ndarray
#             Cleaned binary image
#         """
#         # Remove small objects
#         binary_cleaned = morphology.remove_small_objects(
#             binary_image, min_size=self.min_area
#         )

#         # Fill small holes
#         binary_cleaned = morphology.remove_small_holes(
#             binary_cleaned, area_threshold=self.min_area
#         )

#         # Morphological closing to connect nearby regions
#         selem = disk(self.morphology_disk_size)
#         binary_cleaned = morphology.binary_closing(binary_cleaned, selem)

#         # Keep only largest connected component (assume nuclear envelope)
#         labeled_image = measure.label(binary_cleaned)
#         if labeled_image.max() == 0:
#             return binary_cleaned

#         # Find largest component
#         props = measure.regionprops(labeled_image)
#         largest_area = max(prop.area for prop in props)

#         # Keep only components within reasonable size range
#         valid_labels = []
#         for prop in props:
#             if self.min_area <= prop.area <= min(self.max_area, largest_area):
#                 valid_labels.append(prop.label)

#         # Create final mask
#         final_mask = np.zeros_like(binary_cleaned, dtype=bool)
#         for label in valid_labels:
#             final_mask |= labeled_image == label

#         return final_mask

#     def extract_boundary_coordinates(self, binary_mask, z_index, method="contour"):
#         """
#         Extract boundary coordinates from binary mask.

#         Parameters:
#         -----------
#         binary_mask : numpy.ndarray
#             Binary mask where True indicates inside nuclear envelope
#         z_index : int
#             Z-slice index
#         method : str
#             Method for boundary extraction ('contour', 'canny', 'marching_squares')

#         Returns:
#         --------
#         numpy.ndarray
#             Array of boundary coordinates with shape (N, 3) for (x, y, z)
#         """
#         if method == "contour":
#             return self._extract_boundary_contour(binary_mask, z_index)
#         elif method == "canny":
#             return self._extract_boundary_canny(binary_mask, z_index)
#         elif method == "marching_squares":
#             return self._extract_boundary_marching_squares(binary_mask, z_index)
#         else:
#             raise ValueError(f"Unknown method: {method}")

#     def _extract_boundary_contour(self, binary_mask, z_index):
#         """Extract boundary using OpenCV contour detection."""
#         # Convert to uint8
#         mask_uint8 = (binary_mask * 255).astype(np.uint8)

#         # Find contours
#         contours, _ = cv2.findContours(
#             mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
#         )

#         boundary_points = []
#         for contour in contours:
#             # Convert contour to (x, y, z) coordinates
#             for point in contour:
#                 x, y = point[0]
#                 boundary_points.append([x, y, z_index])

#         return np.array(boundary_points) if boundary_points else np.empty((0, 3))

#     def _extract_boundary_canny(self, binary_mask, z_index):
#         """Extract boundary using Canny edge detection."""
#         # Convert to float for edge detection
#         mask_float = binary_mask.astype(np.float32)

#         # Apply Canny edge detection
#         edges = canny(
#             mask_float,
#             sigma=self.canny_sigma,
#             low_threshold=self.canny_low_threshold,
#             high_threshold=self.canny_high_threshold,
#         )

#         # Get edge coordinates
#         y_coords, x_coords = np.where(edges)
#         z_coords = np.full_like(x_coords, z_index)

#         boundary_points = np.column_stack([x_coords, y_coords, z_coords])
#         return boundary_points

#     def _extract_boundary_marching_squares(self, binary_mask, z_index):
#         """Extract boundary using marching squares algorithm."""
#         try:
#             # Use marching squares to find contours
#             contours = measure.find_contours(binary_mask.astype(np.float32), level=0.5)

#             boundary_points = []
#             for contour in contours:
#                 # contour is in (row, col) format, convert to (x, y, z)
#                 for point in contour:
#                     y, x = point  # row, col -> y, x
#                     boundary_points.append([x, y, z_index])

#             return np.array(boundary_points) if boundary_points else np.empty((0, 3))

#         except Exception as e:
#             print(f"Warning: Marching squares failed for z={z_index}: {e}")
#             return np.empty((0, 3))

#     def process_stack(self, stack, boundary_method="contour", progress_callback=None):
#         """
#         Process entire z-stack to extract boundaries.

#         Parameters:
#         -----------
#         stack : numpy.ndarray
#             3D array with shape (z, y, x)
#         boundary_method : str
#             Method for boundary extraction
#         progress_callback : callable, optional
#             Function to call with progress updates

#         Returns:
#         --------
#         tuple
#             (segmented_stack, boundary_points)
#             - segmented_stack: 3D binary array of segmentations
#             - boundary_points: numpy array of all boundary coordinates
#         """
#         z_slices, height, width = stack.shape

#         # Initialize results
#         segmented_stack = np.zeros((z_slices, height, width), dtype=bool)
#         all_boundary_points = []

#         print(f"Processing {z_slices} z-slices...")

#         for z in range(z_slices):
#             if progress_callback:
#                 progress_callback(z, z_slices)
#             else:
#                 print(f"Processing slice {z+1}/{z_slices}", end="\r")

#             # Preprocess slice
#             preprocessed = self.preprocess_slice(stack[z])

#             # Segment nuclear envelope
#             binary_mask = self.segment_nuclear_envelope(preprocessed)
#             segmented_stack[z] = binary_mask

#             # Extract boundary coordinates
#             boundary_coords = self.extract_boundary_coordinates(
#                 binary_mask, z, method=boundary_method
#             )

#             if len(boundary_coords) > 0:
#                 all_boundary_points.append(boundary_coords)

#         # Combine all boundary points
#         if all_boundary_points:
#             boundary_points = np.vstack(all_boundary_points)
#         else:
#             boundary_points = np.empty((0, 3))

#         print(f"\nCompleted! Extracted {len(boundary_points)} boundary points.")

#         return segmented_stack, boundary_points

#     def construct_point_cloud(
#         self, boundary_points, z_scale=1.0, xy_scale=1.0, remove_outliers=True
#     ):
#         """
#         Construct and optionally clean 3D point cloud.

#         Parameters:
#         -----------
#         boundary_points : numpy.ndarray
#             Array of boundary coordinates with shape (N, 3)
#         z_scale : float
#             Scaling factor for z-coordinates (to account for slice spacing)
#         xy_scale : float
#             Scaling factor for xy-coordinates (to account for pixel size)
#         remove_outliers : bool
#             Whether to remove statistical outliers

#         Returns:
#         --------
#         numpy.ndarray
#             Cleaned and scaled point cloud
#         """
#         if len(boundary_points) == 0:
#             return boundary_points

#         # Scale coordinates
#         scaled_points = boundary_points.copy().astype(np.float32)
#         scaled_points[:, 0] *= xy_scale  # x
#         scaled_points[:, 1] *= xy_scale  # y
#         scaled_points[:, 2] *= z_scale  # z

#         if remove_outliers:
#             scaled_points = self._remove_outliers(scaled_points)

#         return scaled_points

#     def _remove_outliers(self, points, std_threshold=2.0):
#         """
#         Remove statistical outliers from point cloud.

#         Parameters:
#         -----------
#         points : numpy.ndarray
#             Point cloud array
#         std_threshold : float
#             Standard deviation threshold for outlier removal

#         Returns:
#         --------
#         numpy.ndarray
#             Point cloud with outliers removed
#         """
#         if len(points) < 10:  # Too few points to meaningfully remove outliers
#             return points

#         # Calculate distances to centroid for each point
#         centroid = np.mean(points, axis=0)
#         distances = np.linalg.norm(points - centroid, axis=1)

#         # Remove points beyond threshold
#         mean_dist = np.mean(distances)
#         std_dist = np.std(distances)
#         threshold = mean_dist + std_threshold * std_dist

#         valid_points = points[distances <= threshold]

#         print(f"Removed {len(points) - len(valid_points)} outlier points")

#         return valid_points

#     def visualize_results(
#         self,
#         stack,
#         segmented_stack,
#         boundary_points,
#         z_slice_to_show=None,
#         save_path=None,
#     ):
#         """
#         Visualize processing results.

#         Parameters:
#         -----------
#         stack : numpy.ndarray
#             Original z-stack
#         segmented_stack : numpy.ndarray
#             Segmented z-stack
#         boundary_points : numpy.ndarray
#             3D boundary points
#         z_slice_to_show : int, optional
#             Specific z-slice to show (default: middle slice)
#         save_path : str, optional
#             Path to save the visualization
#         """
#         if z_slice_to_show is None:
#             z_slice_to_show = stack.shape[0] // 2

#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))

#         # Original slice
#         axes[0, 0].imshow(stack[z_slice_to_show], cmap="gray")
#         axes[0, 0].set_title(f"Original Z-slice {z_slice_to_show}")
#         axes[0, 0].axis("off")

#         # Segmented slice
#         axes[0, 1].imshow(segmented_stack[z_slice_to_show], cmap="gray")
#         axes[0, 1].set_title(f"Segmented Z-slice {z_slice_to_show}")
#         axes[0, 1].axis("off")

#         # Overlay
#         axes[0, 2].imshow(stack[z_slice_to_show], cmap="gray", alpha=0.7)
#         axes[0, 2].imshow(segmented_stack[z_slice_to_show], cmap="Reds", alpha=0.3)
#         axes[0, 2].set_title(f"Overlay Z-slice {z_slice_to_show}")
#         axes[0, 2].axis("off")

#         # 3D point cloud projections
#         if len(boundary_points) > 0:
#             # XY projection
#             axes[1, 0].scatter(
#                 boundary_points[:, 0],
#                 boundary_points[:, 1],
#                 c=boundary_points[:, 2],
#                 cmap="viridis",
#                 s=1,
#             )
#             axes[1, 0].set_title("XY Projection (colored by Z)")
#             axes[1, 0].set_xlabel("X")
#             axes[1, 0].set_ylabel("Y")

#             # XZ projection
#             axes[1, 1].scatter(
#                 boundary_points[:, 0],
#                 boundary_points[:, 2],
#                 c=boundary_points[:, 1],
#                 cmap="viridis",
#                 s=1,
#             )
#             axes[1, 1].set_title("XZ Projection (colored by Y)")
#             axes[1, 1].set_xlabel("X")
#             axes[1, 1].set_ylabel("Z")

#             # YZ projection
#             axes[1, 2].scatter(
#                 boundary_points[:, 1],
#                 boundary_points[:, 2],
#                 c=boundary_points[:, 0],
#                 cmap="viridis",
#                 s=1,
#             )
#             axes[1, 2].set_title("YZ Projection (colored by X)")
#             axes[1, 2].set_xlabel("Y")
#             axes[1, 2].set_ylabel("Z")
#         else:
#             for i in range(3):
#                 axes[1, i].text(
#                     0.5,
#                     0.5,
#                     "No boundary points found",
#                     ha="center",
#                     va="center",
#                     transform=axes[1, i].transAxes,
#                 )
#                 axes[1, i].set_title(f"Projection {i+1}")

#         plt.tight_layout()

#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches="tight")

#         plt.show()

#     def save_point_cloud(self, boundary_points, filepath, format="npy"):
#         """
#         Save point cloud to file.

#         Parameters:
#         -----------
#         boundary_points : numpy.ndarray
#             3D boundary points
#         filepath : str or Path
#             Output file path
#         format : str
#             Output format ('npy', 'csv', 'txt')
#         """
#         filepath = Path(filepath)

#         if format == "npy":
#             np.save(filepath, boundary_points)
#         elif format == "csv":
#             np.savetxt(
#                 filepath, boundary_points, delimiter=",", header="x,y,z", comments=""
#             )
#         elif format == "txt":
#             np.savetxt(
#                 filepath, boundary_points, delimiter=" ", header="x y z", comments=""
#             )
#         else:
#             raise ValueError(f"Unsupported format: {format}")

#         print(f"Point cloud saved to: {filepath}")


# def process_nuclear_envelope_files(
#     input_dir, output_dir=None, file_pattern="*.tif*", **processor_kwargs
# ):
#     """
#     Process multiple nuclear envelope TIFF files.

#     Parameters:
#     -----------
#     input_dir : str or Path
#         Directory containing TIFF files
#     output_dir : str or Path, optional
#         Directory to save results (default: input_dir/results)
#     file_pattern : str
#         File pattern to match (default: "*.tif*")
#     **processor_kwargs : dict
#         Additional arguments for NuclearEnvelopeProcessor

#     Returns:
#     --------
#     dict
#         Dictionary mapping filenames to their results
#     """
#     input_dir = Path(input_dir)
#     if output_dir is None:
#         output_dir = input_dir / "results"
#     else:
#         output_dir = Path(output_dir)

#     output_dir.mkdir(exist_ok=True)

#     # Find TIFF files
#     tiff_files = list(input_dir.glob(file_pattern))
#     if not tiff_files:
#         print(f"No TIFF files found in {input_dir} matching pattern {file_pattern}")
#         return {}

#     print(f"Found {len(tiff_files)} TIFF files to process")

#     # Initialize processor
#     processor = NuclearEnvelopeProcessor(**processor_kwargs)

#     results = {}

#     for i, tiff_file in enumerate(tiff_files):
#         print(f"\n{'='*60}")
#         print(f"Processing file {i+1}/{len(tiff_files)}: {tiff_file.name}")
#         print("=" * 60)

#         try:
#             # Load stack
#             stack = processor.load_tiff_stack(tiff_file)

#             # Process stack
#             segmented_stack, boundary_points = processor.process_stack(stack)

#             # Construct point cloud
#             point_cloud = processor.construct_point_cloud(boundary_points)

#             # Save results
#             base_name = tiff_file.stem

#             # Save point cloud
#             pc_file = output_dir / f"{base_name}_point_cloud.npy"
#             processor.save_point_cloud(point_cloud, pc_file, format="npy")

#             # Save as CSV too
#             csv_file = output_dir / f"{base_name}_point_cloud.csv"
#             processor.save_point_cloud(point_cloud, csv_file, format="csv")

#             # Save segmentation stack
#             seg_file = output_dir / f"{base_name}_segmentation.npy"
#             np.save(seg_file, segmented_stack)

#             # Create and save visualization
#             viz_file = output_dir / f"{base_name}_visualization.png"
#             processor.visualize_results(
#                 stack, segmented_stack, point_cloud, save_path=viz_file
#             )
#             plt.close("all")  # Close figures to save memory

#             # Store results
#             results[tiff_file.name] = {
#                 "point_cloud": point_cloud,
#                 "segmented_stack": segmented_stack,
#                 "num_boundary_points": len(point_cloud),
#                 "stack_shape": stack.shape,
#             }

#             print(f"Successfully processed {tiff_file.name}")
#             print(f"  - Extracted {len(point_cloud)} boundary points")
#             print(f"  - Stack shape: {stack.shape}")

#         except Exception as e:
#             print(f"Error processing {tiff_file.name}: {e}")
#             results[tiff_file.name] = {"error": str(e)}

#     return results


# # Example usage and demonstration
# if __name__ == "__main__":
#     # Example: Process a single file
#     """
#     # Initialize processor with custom parameters
#     processor = NuclearEnvelopeProcessor(
#         gaussian_sigma=1.5,
#         min_area=200,
#         max_area=8000,
#         canny_sigma=1.0,
#         morphology_disk_size=2
#     )

#     # Load and process a single TIFF stack
#     try:
#         stack = processor.load_tiff_stack("path/to/your/nuclear_envelope.tif")
#         segmented_stack, boundary_points = processor.process_stack(stack)
#         point_cloud = processor.construct_point_cloud(boundary_points)

#         # Visualize results
#         processor.visualize_results(stack, segmented_stack, point_cloud)

#         # Save point cloud
#         processor.save_point_cloud(point_cloud, "nuclear_envelope_pointcloud.npy")

#         print(f"Point cloud shape: {point_cloud.shape}")
#         print(f"Point cloud statistics:")
#         print(f"  X range: {point_cloud[:, 0].min():.2f} - {point_cloud[:, 0].max():.2f}")
#         print(f"  Y range: {point_cloud[:, 1].min():.2f} - {point_cloud[:, 1].max():.2f}")
#         print(f"  Z range: {point_cloud[:, 2].min():.2f} - {point_cloud[:, 2].max():.2f}")

#     except FileNotFoundError:
#         print("Please provide the path to your TIFF file")
#     """

#     # Example: Process multiple files in a directory
#     """
#     results = process_nuclear_envelope_files(
#         input_dir="path/to/tiff/directory",
#         output_dir="path/to/output/directory",
#         gaussian_sigma=1.5,
#         min_area=200
#     )

#     # Print summary
#     total_points = sum(r.get('num_boundary_points', 0) for r in results.values()
#                       if 'error' not in r)
#     successful = sum(1 for r in results.values() if 'error' not in r)

#     print(f"\nProcessing Summary:")
#     print(f"  Files processed successfully: {successful}/{len(results)}")
#     print(f"  Total boundary points extracted: {total_points}")
#     """

#     print("Nuclear Envelope Processor ready!")
#     print("\nTo use this script:")
#     print("1. Uncomment and modify the example usage code above")
#     print("2. Provide the path to your TIFF files")
#     print("3. Run the script")
#     print("\nThe script will:")
#     print("  - Segment nuclear envelope in each z-slice")
#     print("  - Extract boundary coordinates")
#     print("  - Construct 3D point cloud")
#     print("  - Save results as NumPy arrays and CSV files")
#     print("  - Generate visualization plots")

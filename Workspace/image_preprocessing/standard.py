import numpy as np
import cv2
from skimage.segmentation import felzenszwalb, relabel_sequential
from graph_creation.delauny_graph import delauny_graph_from_superpixels
from torch_geometric.utils.convert import from_networkx


def image_to_graph(image, threshold_value=0.5, segmentation_params=None):
    """
    Process a skin lesion image to produce a graph for GNN training.

    Pipeline:
      1. Normalize the image to float32 in [0,1].
      2. Convert the image to grayscale and apply a fixed binary threshold to
         isolate the lesion (non-lesion areas become zero).
      3. Mask the original image with the binary mask.
      4. Convert the masked image to uint8 (scaling to 0-255) for segmentation.
      5. Perform Felzenszwalb segmentation on the masked image.
      6. Remove segments that are entirely background (all pixel values zero).
      7. Relabel the remaining segments to ensure contiguous labels.
      8. Create a region adjacency graph (RAG) from the filtered segments.

    Parameters:
      image: Input RGB image as a numpy array normalized to [0,1].
      threshold_value: Threshold in [0,1] used to create the binary mask.
      segmentation_params: Optional dictionary of parameters for Felzenszwalb,
                           e.g., {'scale': 100, 'sigma': 0.5, 'min_size': 50}.

    Returns:
      A graph object representing the image segmentation.
    """
    # Ensure the image is float32 in [0,1].
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Step 1: Convert to grayscale (scaled to [0,255] for cv2) for thresholding.
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Step 2: Apply a fixed binary threshold.
    # The threshold value is converted from [0,1] to [0,255].
    thresh_uint8 = int(threshold_value * 255)
    _, binary_mask = cv2.threshold(gray, thresh_uint8, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.float32)

    # Step 3: Mask the original image so that only the lesion remains.
    masked_image = image * binary_mask[:, :, np.newaxis]

    # Step 4: Convert the masked image to uint8 for segmentation.
    masked_image_uint8 = (masked_image * 255).astype(np.uint8)

    # Step 5: Segment the masked image using Felzenszwalb.
    if segmentation_params is None:
        segmentation_params = {"scale": 100, "sigma": 0.5, "min_size": 50}
    segments = felzenszwalb(
        masked_image_uint8,
        scale=segmentation_params.get("scale", 100),
        sigma=segmentation_params.get("sigma", 0.5),
        min_size=segmentation_params.get("min_size", 50),
    )

    # Step 6: Filter out segments that are entirely background.
    # For each segment, check if all corresponding pixels in the masked image are zero.
    filtered_segments = np.zeros_like(segments)
    current_label = 1
    for seg in np.unique(segments):
        seg_mask = segments == seg
        # If all pixels in the segment are zero, skip this segment.
        if np.all(masked_image[seg_mask] == 0):
            continue
        else:
            filtered_segments[seg_mask] = current_label
            current_label += 1

    # Step 7: Relabel segments to ensure contiguous labels starting from 1.
    filtered_segments, _, _ = relabel_sequential(filtered_segments)

    # Step 8: Create a graph from the filtered segments.
    # We pass the original image (converted to uint8) for computing mean color differences.
    G = delauny_graph_from_superpixels(filtered_segments, image)  # Not (image * 255)
    return from_networkx(G, group_node_attrs=["x"], group_edge_attrs=["edge_attr"])

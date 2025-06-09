import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature import peak_local_max
from skimage.segmentation import slic, felzenszwalb, watershed
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
from scipy import ndimage as ndi

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

def stratified_split(dataset, test_size=0.2, random_state=0):
    """
    Perform stratified random split on a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The full dataset object.
        y (array-like or torch.Tensor): The labels corresponding to each sample in the dataset.
        test_size (float): Fraction of the dataset to use as the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_subset (torch.utils.data.Subset): Stratified training subset.
        test_subset (torch.utils.data.Subset): Stratified test subset.
    """
    y = dataset.labels
    if hasattr(y, 'numpy'):
        y = y.numpy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    indices = list(range(len(dataset)))
    
    train_idx, test_idx = next(sss.split(indices, y))
    
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    
    return train_subset, test_subset


def create_superpixels(img, method='slic', **kwargs):
    gray = rgb2gray(img)
    if method == 'slic':
        kwargs["n_segments"] = 100
        return slic(img, **kwargs)
    elif method == 'felzenszwalb':
        return felzenszwalb(img, **kwargs)
    elif method == 'watershed':
        gradient = sobel(gray)
        
        distance = ndi.distance_transform_edt(gray)
        
        coords = peak_local_max(
            distance,
            min_distance=int(0.1*min(distance.shape)),
            threshold_rel=0.3,
            footprint=np.ones((3, 3)),
            labels=(gray > 0.1*gray.mean())
        )
        
        markers = np.zeros(gray.shape, dtype=np.int32)
        markers[tuple(coords.T)] = 1
        markers = ndi.label(markers)[0]

        return watershed(
            -gradient,
            markers,
            mask=(gray > 0.1*gray.mean()),
            watershed_line=True,
            compactness=0.01
        )
    elif method == 'kmeans':
        flat = img.reshape(-1, img.shape[2])
        kmeans = KMeans(n_clusters=kwargs.get('n_segments', 100)).fit(flat)
        return kmeans.labels_.reshape(img.shape[:2])
    elif method == 'pixel':
        h, w = gray.shape
        return np.arange(h * w).reshape(h, w)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def superpixel_properties(img, segments):
    """Compute mean, median, and centroid for each label in `segments`."""
    props = {}
    coords = {}
    for label in np.unique(segments):
        mask = segments == label
        values = img[mask]
        props[label] = {
            'mean': values.mean(),
            'median': np.median(values),
            # also could compute var, skewness, etc.
        }
        ys, xs = np.where(mask)
        coords[label] = {
            'centroid_y': ys.mean(),
            'centroid_x': xs.mean(),
            # optionally bounding box: (ys.min(), xs.min(), ys.max(), xs.max())
        }
    return props, coords


def compute_texture(img, segments, P=8, R=1):
    """
    Compute the mean LBP value per superpixel.
    - P: number of circularly symmetric neighbour set points.
    - R: radius.
    Returns a dict: {label: mean_lbp}.
    """
    gray = rgb2gray(img)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    texture = {}
    for label in np.unique(segments):
        mask = segments == label
        texture[label] = lbp[mask].mean()
    return texture


def compute_shape_metrics(segments):
    """
    For each labeled region, compute:
      - area (pixel count)
      - perimeter
      - compactness = perimeter^2 / area
    Returns dict of dicts: {label: {'area':…, 'perimeter':…, 'compactness':…}}.
    """
    props = regionprops(segments.astype(int))
    shape = {}
    for p in props:
        label = p.label
        area = p.area
        perim = p.perimeter
        compact = (perim ** 2) / area if area > 0 else 0
        shape[label] = {
            'area': area,
            'perimeter': perim,
            'compactness': compact
        }
    return shape

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature import peak_local_max
from skimage.segmentation import slic, felzenszwalb
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops


def create_superpixels(img, method='slic', **kwargs):
    gray = rgb2gray(img)
    if method == 'slic':
        return slic(img, **kwargs)
    elif method == 'felzenszwalb':
        return felzenszwalb(img, **kwargs)
    elif method == 'watershed':
        elevation = sobel(gray)
        markers = peak_local_max(elevation, indices=False, footprint=np.ones((3, 3)))
        from skimage.morphology import watershed as sk_watershed
        return sk_watershed(elevation, markers, **kwargs)
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

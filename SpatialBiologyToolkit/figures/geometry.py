"""Shared per-object geometry, preserving adjacent objects and holes."""
import numpy as np
from skimage.util import map_array
from skimage.measure import find_contours

def _population_cell_paths(mask, target_cell_ids):
    """Yield one compound path per selected label, preserving touching cells/holes.

    Dense temporary labels avoid allocations proportional to potentially sparse
    ObjectNumbers; contours are traced only inside each cell's bounding box.
    Padding closes contours for cells touching the image boundary.
    """
    from scipy.ndimage import find_objects
    from matplotlib.path import Path as MplPath

    ids = np.intersect1d(np.unique(mask), list(target_cell_ids))
    ids = ids[ids != 0].astype(mask.dtype)
    if not len(ids):
        return
    labels = map_array(mask, ids, np.arange(1, len(ids) + 1, dtype=np.int32))
    for label, slices in enumerate(find_objects(labels), start=1):
        if slices is None:
            continue
        local = np.pad(labels[slices] == label, 1)
        paths = []
        for contour in find_contours(local, 0.5):
            vertices = contour[:, ::-1] + [slices[1].start - 1, slices[0].start - 1]
            paths.append(MplPath(vertices, closed=True))
        if paths:
            yield int(ids[label - 1]), MplPath.make_compound_path(*paths)



import numpy as np

VOC_COLORMAP = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
])
def rgb_to_label(mask: np.ndarray, colormap : np.ndarray=VOC_COLORMAP) -> np.ndarray:
    """
    Convert an RGB mask to a label mask using a colormap.

    Parameters:
    mask (np.ndarray): RGB mask as a NumPy array.
    colormap (np.ndarray): Colormap for converting RGB to labels.

    Returns:
    np.ndarray: Label mask as a NumPy array.
    """
    mask = np.array(mask)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
    for i, color in enumerate(colormap):
        label_mask[np.all(mask == color, axis=-1)] = i
    return label_mask
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

IMG_SIZE = 224
NUM_CLASSES = 21

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

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    np.ndarray: Preprocessed image as a NumPy array.
    """
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image /= 255.0  # Normalize to [0, 1]
    return image

def rgb_to_label(mask: np.ndarray, colormap: np.ndarray) -> np.ndarray:
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

def preprocess_mask(mask_path: str) -> tf.Tensor:
    """
    Load and preprocess a mask.

    Parameters:
    mask_path (str): Path to the mask file.

    Returns:
    tf.Tensor: Preprocessed mask as a TensorFlow tensor.
    """
    mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(IMG_SIZE, IMG_SIZE))
    mask = tf.keras.preprocessing.image.img_to_array(mask)
    label_mask = rgb_to_label(mask, VOC_COLORMAP)
    return tf.expand_dims(label_mask, axis=-1)

def load_and_preprocess_data(image_dir: str, mask_dir: str) -> tuple:
    """
    Load and preprocess images and masks, splitting them into training and validation sets.

    Parameters:
    image_dir (str): Directory containing images.
    mask_dir (str): Directory containing masks.

    Returns:
    tuple: Four lists containing preprocessed training images, validation images,
           training masks, and validation masks.
    """
    images = []
    masks = []
    for image_filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_filename)
        mask_path = os.path.join(mask_dir, image_filename.replace('.jpg', '.png'))
        if os.path.exists(mask_path):
            images.append(preprocess_image(image_path))
            masks.append(preprocess_mask(mask_path))
    return train_test_split(images, masks, test_size=0.2, random_state=42)


def create_tf_dataset(images: list, masks: list) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from images and masks.

    Parameters:
    images (list): List of preprocessed images.
    masks (list): List of preprocessed masks.

    Returns:
    tf.data.Dataset: TensorFlow dataset containing images and masks.
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(len(images)).batch(8).prefetch(tf.data.AUTOTUNE)
    return dataset
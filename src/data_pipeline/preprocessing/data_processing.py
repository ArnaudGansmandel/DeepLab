import os
from data_pipeline.preprocessing.encoder import rgb_to_label
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

IMG_SIZE = 224
NUM_CLASSES = 21

# src/data_pipeline/preprocessing.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

class Preprocessor:
    def __init__(self, img_size=224):
        self.img_size = img_size

    def preprocess_image(self, image_path: str) -> tf.Tensor:
        """
        Load and preprocess an image.

        Parameters:
        image_path (str): Path to the image file.

        Returns:
        tf.Tensor: Preprocessed image as a TensorFlow tensor.
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.img_size, self.img_size])
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        return image

    def preprocess_mask(self, mask_path: str, num_classes: int) -> tf.Tensor:
        """
        Load and preprocess a mask.

        Parameters:
        mask_path (str): Path to the mask file.
        num_classes (int): Number of classes for the segmentation.

        Returns:
        tf.Tensor: Preprocessed mask as a TensorFlow tensor.
        """
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)  # Assuming masks are in PNG format
        mask = tf.image.resize(mask, [self.img_size, self.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = self.rgb_to_label(mask, num_classes)
        return tf.expand_dims(mask, axis=-1)

    def rgb_to_label(self, mask: tf.Tensor, num_classes: int) -> tf.Tensor:
        """
        Convert RGB mask to class labels.

        Parameters:
        mask (tf.Tensor): Input RGB mask.
        num_classes (int): Number of classes for segmentation.

        Returns:
        tf.Tensor: Mask with class labels.
        """
        # Your color to label conversion logic here
        # Example for VOC:
        VOC_COLORMAP = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                                 [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                                 [0, 192, 0], [128, 192, 0], [0, 64, 128]])
        label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for i, col in enumerate(VOC_COLORMAP):
            label_mask[np.all(mask == col, axis=-1)] = i
        return tf.convert_to_tensor(label_mask, dtype=tf.uint8)

    
############################################################
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
    label_mask = rgb_to_label(mask)
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
import os
from data_pipeline.preprocessing.encoder import rgb_to_label
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMG_SIZE = 224

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
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

class Preprocessor:
    def __init__(self, config):
        self.img_size = config['img_size']

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

    def preprocess_mask(self, mask_path: str) -> tf.Tensor:
        """
        Load and preprocess a mask.

        Parameters:
        mask_path (str): Path to the mask file.

        Returns:
        tf.Tensor: Preprocessed mask as a TensorFlow tensor.
        """
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)  # Assuming masks are in PNG format
        mask = tf.image.resize(mask, [self.img_size, self.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = self.rgb_to_label(mask)
        return tf.expand_dims(mask, axis=-1)

    def rgb_to_label(self, mask: tf.Tensor) -> tf.Tensor:
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
            mask_equal = tf.reduce_all(tf.equal(mask, col), axis=-1)
            label_mask = tf.where(mask_equal, i, label_mask)

        return label_mask
    
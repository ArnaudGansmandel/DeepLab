# src/data_pipeline/augmentation.py

import tensorflow as tf
from tensorflow.keras import layers

class DataAugmentation:
    def __init__(self, config):
        self.img_size = config['img_size']
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])

    def augment(self, image: tf.Tensor, mask: tf.Tensor) -> (tf.Tensor, tf.Tensor): # type: ignore
        """
        Apply data augmentation to an image and its corresponding mask.

        Parameters:
        image (tf.Tensor): Input image.
        mask (tf.Tensor): Corresponding mask.

        Returns:
        tuple: Augmented image and mask.
        """
        # Apply augmentation to both image and mask
        image = self.data_augmentation(image)
        mask = self.data_augmentation(mask)

        return image, mask

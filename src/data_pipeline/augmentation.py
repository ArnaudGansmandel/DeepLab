import tensorflow as tf
from tensorflow.keras import layers

class DataAugmentation:
    def __init__(self):
        # Store augmentation pipelines in a list
        self.augmentations = [
            tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
            ]),
            tf.keras.Sequential([
                layers.Resizing(448, 448),
                layers.RandomCrop(224, 224)
            ]),
            tf.keras.Sequential([
                layers.Resizing(672, 672),
                layers.RandomCrop(224, 224)
            ]),
            tf.keras.Sequential([
                layers.Resizing(896, 896),
                layers.RandomCrop(224, 224)
            ])
        ]



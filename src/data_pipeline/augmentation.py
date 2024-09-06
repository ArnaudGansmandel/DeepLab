# src\data_pipeline\augmentation.py
import tensorflow as tf

class DataAugmentation:
    def __init__(self):
        self.augmentations = [
            self.flip_and_rotate
            # self.resize_and_crop_448,
            # self.resize_and_crop_672,
            # self.resize_and_crop_896,
        ]

    def flip_and_rotate(self, image, mask):
        # Apply the same random flip to both image and mask
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)

        # Apply the same random rotation to both image and mask
        rotation_angle = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=rotation_angle)
        mask = tf.image.rot90(mask, k=rotation_angle)

        return image, mask

    def resize_and_crop(self, image, mask, size):
        # Resize both image and mask
        image = tf.image.resize(image, [size, size])
        mask = tf.image.resize(mask, [size, size], method='nearest')

        # Random crop both image and mask to 224x224
        image_mask_combined = tf.concat([image, mask], axis=-1)
        cropped_combined = tf.image.random_crop(image_mask_combined, size=[224, 224, tf.shape(image)[-1] + tf.shape(mask)[-1]])

        image = cropped_combined[..., :tf.shape(image)[-1]]
        mask = cropped_combined[..., tf.shape(image)[-1]:]

        return image, mask

    def resize_and_crop_448(self, image, mask):
        return self.resize_and_crop(image, mask, 448)

    def resize_and_crop_672(self, image, mask):
        return self.resize_and_crop(image, mask, 672)

    def resize_and_crop_896(self, image, mask):
        return self.resize_and_crop(image, mask, 896)




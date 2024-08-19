import os

import tensorflow as tf


class DataLoader:
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def load_data(self):
        images = []
        masks = []
        for image_filename in os.listdir(self.image_dir):
            image_path = os.path.join(self.image_dir, image_filename)
            mask_path = os.path.join(self.mask_dir, image_filename.replace('.jpg', '.png'))
            if os.path.exists(mask_path):
                images.append(image_path)
                masks.append(mask_path)
        return images, masks
    
    def create_tf_dataset(self, images, masks):
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        dataset = dataset.shuffle(len(images)).batch(8).prefetch(tf.data.AUTOTUNE)
        return dataset
    
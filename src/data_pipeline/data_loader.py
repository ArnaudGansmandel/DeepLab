import os
import tensorflow as tf
from data_pipeline.preprocessing import Preprocessor 
from data_pipeline.augmentation import DataAugmentation

class DataLoader:
    def __init__(self, dir_path, config):
        self.dir_path = dir_path
        self.config = config
        self.img_size = config['img_size']
        self.batch_size = config['batch_size']
        self.preprocessor = Preprocessor(img_size=self.img_size)
        self.augmentor = DataAugmentation(img_size=self.img_size)
        self.image_dir = os.path.join(dir_path, 'images')
        self.mask_dir = os.path.join(dir_path, 'masks')

        self.image_list_ds = tf.data.Dataset.list_files(os.path.join(self.image_dir, '*/*'), shuffle=False)
        self.mask_list_ds = tf.data.Dataset.list_files(os.path.join(self.mask_dir, '*/*'), shuffle=False)

    def __configure_for_performance(self, ds, shuffle=True):
        AUTOTUNE = tf.data.AUTOTUNE
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def load_data(self, augment=False):
        image_dataset = self.image_list_ds.map(lambda x: self.preprocessor.preprocess_image(x),
                                               num_parallel_calls=tf.data.AUTOTUNE)
        mask_dataset = self.mask_list_ds.map(lambda x: self.preprocessor.preprocess_mask(x, self.config['num_classes']),
                                             num_parallel_calls=tf.data.AUTOTUNE)
        dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
        dataset = self.__configure_for_performance(dataset)

        if augment:
            dataset = dataset.map(lambda x, y: self.augmentor.augment(x, y),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

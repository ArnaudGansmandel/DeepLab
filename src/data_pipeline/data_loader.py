import os
import pathlib
import tensorflow as tf
from data_pipeline.preprocessing.data_processing import Preprocessor
from data_pipeline.augmentation import DataAugmentation

class DataLoader:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.preprocessor = Preprocessor(config)
        self.augmentor = DataAugmentation(config)

    def create_file_dataset(self, dir_path):
            image_dir = pathlib.Path(os.path.join('data', dir_path, 'images'))
            mask_dir = pathlib.Path(os.path.join('data', dir_path, 'masks'))
            image_list_ds = tf.data.Dataset.list_files(str(image_dir/'*'))
            mask_list_ds = tf.data.Dataset.list_files(str(mask_dir/'*'))

            return image_list_ds, mask_list_ds

    def __configure_for_performance(self, ds, shuffle=True):
        AUTOTUNE = tf.data.AUTOTUNE
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def load_data(self, dir_path, augment=False):
        image_list_ds, mask_list_ds = self.create_file_dataset(dir_path)
        image_dataset = image_list_ds.map(lambda x: self.preprocessor.preprocess_image(x),
                                               num_parallel_calls=tf.data.AUTOTUNE)
        mask_dataset = mask_list_ds.map(lambda x: self.preprocessor.preprocess_mask(x),
                                             num_parallel_calls=tf.data.AUTOTUNE)
        dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))

        if augment:
            dataset = dataset.map(lambda x, y: self.augmentor.augment(x, y),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = self.__configure_for_performance(dataset)

        return dataset

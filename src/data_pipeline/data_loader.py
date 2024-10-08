# src\data_pipeline\data_loader.py
import os
import pathlib
from matplotlib import pyplot as plt
import tensorflow as tf
from data_pipeline.preprocessing import Preprocessor
from data_pipeline.augmentation import DataAugmentation
from utils.plotting import display

class DataLoader:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.preprocessor = Preprocessor(config)
        self.augmentor = DataAugmentation()

    def create_file_dataset(self, dir_path):
            """
            Create datasets for images and masks based on file paths.

            Parameters:
            dir_path (str): Directory path containing 'images' and 'masks' folders.

            Returns:
            tuple: A tuple of image and mask datasets.
            """
            image_dir = pathlib.Path(os.path.join('data', dir_path, 'images'))
            mask_dir = pathlib.Path(os.path.join('data', dir_path, 'masks'))
            image_list_ds = tf.data.Dataset.list_files(str(image_dir/'*'), shuffle=False)
            mask_list_ds = tf.data.Dataset.list_files(str(mask_dir/'*'), shuffle=False)

            return image_list_ds, mask_list_ds

    def __configure_for_performance(self, ds, shuffle=True):
        """
        Configure the dataset for performance with caching, shuffling, batching, and prefetching.

        Parameters:
        ds (tf.data.Dataset): The dataset to configure.
        shuffle (bool): Whether to shuffle the dataset.

        Returns:
        tf.data.Dataset: The configured dataset.
        """
        AUTOTUNE = tf.data.AUTOTUNE
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def load_data(self, dir_path, augment=False):
        """
        Load the data, apply preprocessing, and optionally apply augmentations.

        Parameters:
        dir_path (str): Directory path containing the data.
        augment (bool): Whether to apply data augmentations.

        Returns:
        tf.data.Dataset: The final dataset ready for training or evaluation.
        """
        image_list_ds, mask_list_ds = self.create_file_dataset(dir_path)
        image_dataset = image_list_ds.map(lambda x: self.preprocessor.preprocess_image(x),
                                               num_parallel_calls=tf.data.AUTOTUNE)
        mask_dataset = mask_list_ds.map(lambda x: self.preprocessor.preprocess_mask(x),
                                             num_parallel_calls=tf.data.AUTOTUNE)
        dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))  
        # for image, mask in dataset.take(1):
        #     display([image, mask])
        if augment:
            augmented_datasets = []
            for aug in self.augmentor.augmentations:
                augmented_dataset = dataset.map(lambda x, y: aug(x, y),
                                                num_parallel_calls=tf.data.AUTOTUNE)

                # for image, mask in augmented_dataset.take(1):
                #     display([image, mask])                

                augmented_datasets.append(augmented_dataset)
            for aug_ds in augmented_datasets:
                dataset = dataset.concatenate(aug_ds)
        
        dataset = self.__configure_for_performance(dataset)

        return dataset
    
# Example usage
if __name__ == "__main__":
    # Training configuration
    config = {
        'learning_rate': 0.007,
        'fine_tuning_learning_rate': 0.001,
        'epochs': 3,
        'batch_size': 16,
        'num_classes': 21,
        'img_size': 224,
        'checkpoint_filepath': 'results/checkpoints/model.weights.h5',
        'model_save_filepath': 'results/models/top_model.weights.h5',
    }

    data_loader = DataLoader(config)
    train_dataset = data_loader.load_data('train', augment=True)
    #val_dataset = data_loader.load_data('val')

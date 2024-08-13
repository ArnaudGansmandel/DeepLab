from deep_lab.data_processing import preprocess_image, preprocess_mask
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


voc_classes = {
    'background': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}


def display_image_and_mask(image_path: str, mask_path: str) -> None:
    """
    Display an image and its corresponding mask before and after preprocessing.

    Parameters:
    image_path (str): Path to the image file.
    mask_path (str): Path to the mask file.

    Returns:
    None
    """
    # Load the original image and mask
    original_image = tf.keras.preprocessing.image.load_img(image_path)
    original_mask = tf.keras.preprocessing.image.load_img(mask_path)
    
    # Preprocess the image and mask
    preprocessed_image = preprocess_image(image_path)
    preprocessed_mask = preprocess_mask(mask_path)

    # Display the original image and mask
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    
    plt.subplot(2, 2, 2)
    plt.title("Original Mask")
    plt.imshow(original_mask)
    
    # Display the preprocessed image and mask
    plt.subplot(2, 2, 3)
    plt.title("Preprocessed Image")
    plt.imshow(preprocessed_image)
    
    plt.subplot(2, 2, 4)
    plt.title("Preprocessed Mask")
    plt.imshow(tf.squeeze(preprocessed_mask), cmap='gray')
    
    plt.show()


import pandas as pd

def create_class_pixel_table(masks: list) -> pd.DataFrame:
    """
    Create a table displaying the number of pixels corresponding to each class.

    Parameters:
    masks (list): List of preprocessed masks.

    Returns:
    pd.DataFrame: DataFrame with class names and corresponding pixel counts.
    """
    # Concatenate all masks to find the unique labels and their counts
    all_masks = np.concatenate(masks)
    labels, counts = np.unique(all_masks, return_counts=True)

    # Create a dictionary for class pixel counts
    pixel_counts = {label: count for label, count in zip(labels, counts)}

    # Map class names to pixel counts
    class_names = list(voc_classes.keys())
    class_labels = list(voc_classes.values())
    class_pixel_counts = [pixel_counts.get(label, 0) for label in class_labels]

    # Create a DataFrame
    data = {'Class': class_names, 'Pixel Count': class_pixel_counts}
    df = pd.DataFrame(data)

    return df



# Faire fonction qui prend en entrée un élément du dataset et affiche une image au hasard en reconvertissant le mask dans son format initiale

# Faire fonction qui retourne summary des models

# Faire la sauvegarde du graph du model
from matplotlib import pyplot as plt
import pandas as pd
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


# fonction display_results that takes a random image (with take) in a dataset and display the original image the mask and the prediction on the same line
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(model, dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])

# Faire fonction qui prend en entrée un élément du dataset et affiche une image au hasard en reconvertissant le mask dans son format initiale

# Faire fonction qui retourne summary des models

# Faire la sauvegarde du graph du model
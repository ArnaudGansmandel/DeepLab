import tensorflow as tf

import tensorflow as tf
import matplotlib.pyplot as plt

def predict_and_display(model, image, original_mask=None):
    # Expand dimensions to add batch size (1, height, width, channels)
    input_image = tf.expand_dims(image, axis=0)
    
    # Get logits from the model
    logits = model.predict(input_image)  # Shape: (1, height, width, num_classes)
    
    # Get the predicted class for each pixel
    predicted_mask = tf.argmax(logits, axis=-1)  # Shape: (1, height, width)
    predicted_mask = tf.squeeze(predicted_mask, axis=0)  # Remove the batch dimension

    # Plot the original image, the ground truth mask (if available), and the predicted mask
    plt.figure(figsize=(15, 5))
    
    # Display the original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    
    # Display the original mask if provided
    if original_mask is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(original_mask, cmap='gray')
        plt.title("Original Mask")

    # Display the predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask.numpy(), cmap='gray')
    plt.title("Predicted Mask")

    plt.show()

# Example usage:
# Assuming `model` is your trained DeepLabV3Plus model
# `sample_image` is a numpy array or tensor of shape (height, width, channels)
# `sample_mask` is the ground truth mask of shape (height, width)

image_dir = r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\DeepLab\images_to_segment\inputs'

deep_lab_model = tf.keras.models.load_model(r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\DeepLab\deep_lab_model')

image_path = r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\DeepLab\images_to_segment\inputs\image_1.jpg'  # Replace with the path to your image file

image = tf.keras.preprocessing.image.load_img(image_path)

sample_image = tf.keras.preprocessing.image.img_to_array(image)

sample_mask = None  # Replace with the path to your mask file if available

predict_and_display(deep_lab_model, sample_image, original_mask=sample_mask)


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

predict_and_display(model, sample_image, original_mask=sample_mask)

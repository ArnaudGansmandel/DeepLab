import tensorflow as tf


class MeanIoU(tf.keras.metrics.Metric):
    """
    Calculates the mean Intersection over Union (IoU) for a given set of classes.

    Args:
        num_classes (int): The number of classes to calculate IoU for.
        name (str): The name of the metric.
        **kwargs: Additional keyword arguments for the metric.
    """
    def __init__(self, num_classes=21, name='mean_iou', **kwargs):
        """
        Initialize the MeanIoU metric.
        """
        super(MeanIoU, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes  # The number of classes to calculate IoU for
        
        # Add a weight to store the total confusion matrix
        # The shape of the confusion matrix is (num_classes, num_classes)
        # The initializer is set to zeros to reset the confusion matrix for each epoch
        self.total_cm = self.add_weight(
            name='total_cm',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the confusion matrix with the current batch of predictions and ground truth.

        Args:
            y_true (tf.Tensor): Ground truth labels.
            y_pred (tf.Tensor): Predicted labels.
            sample_weight (tf.Tensor, optional): Sample weights. Defaults to None.
        """
        # Remove last dimension if necessary
        y_true = tf.squeeze(y_true, axis=-1)
        # Get the predicted labels
        y_pred = tf.argmax(y_pred, axis=-1)

        # Reshape the ground truth and predicted labels to 1-D tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Calculate the confusion matrix for the current batch
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            dtype=tf.float32
        )

        # Add the current confusion matrix to the total confusion matrix
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """
        Calculate the mean IoU by summing the IoU for each class and dividing by the number of classes.
        The IoU is calculated by dividing the true positives by the sum of predicted and true positives
        plus the false negatives.
        """
        # Sum over rows and columns to get the total number of true positives, predicted positives,
        # and true negatives plus predicted negatives for each class.
        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)

        # Get the diagonal of the confusion matrix which contains the true positives for each class.
        true_positives = tf.linalg.diag_part(self.total_cm)

        # Calculate the denominator by summing the predicted positives and true negatives for each class.
        denominator = sum_over_row + sum_over_col - true_positives

        # Count the number of valid entries (classes with at least one true positive or predicted positive).
        num_valid_entries = tf.reduce_sum(tf.cast(denominator > 0, dtype=tf.float32))

        # Calculate the IoU for each class by dividing the true positives by the denominator.
        iou = tf.math.divide_no_nan(true_positives, denominator)

        # Calculate the mean IoU by summing the IoU for each class and dividing by the number of classes.
        return tf.reduce_sum(iou, name=self.name) / num_valid_entries

    def reset_states(self):
        for variable in self.variables:
            tf.keras.backend.set_value(variable, tf.zeros_like(variable))


# src/training/trainer.py
import datetime
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers.schedules import PolynomialDecay

#from training.learning_rate import PolyDecay
from training.metrics import UpdatedMeanIoU
from training.callback import create_callbacks


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        #poly_decay = PolyDecay(initial_learning_rate=config['learning_rate'], max_epochs=config['epochs'])
        poly_decay = PolynomialDecay(initial_learning_rate=config['learning_rate'], decay_steps=2288, power=0.9)
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=poly_decay, weight_decay=config['weight_decay'])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = [
            'accuracy', 
            tf.keras.metrics.SparseCategoricalAccuracy(), 
            UpdatedMeanIoU(num_classes=config['num_classes'])
        ]
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
        
        # Create callbacks
        self.callbacks = create_callbacks(config)

    def train(self):
        """
        Trains the model using the provided training dataset and validation dataset.

        Parameters:
            None

        Returns:
            history (tf.keras.callbacks.History): The training history of the model.
        """
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.config['epochs'],
            callbacks=self.callbacks
        )
        return history
    
    def evaluate(self):
        """
        Evaluates the model on the validation dataset.

        Returns:
            A list of metrics evaluated on the validation dataset.
        """
        results = self.model.evaluate(self.val_dataset)
        return results

    def save_model(self, model_weight_saving_path=None):
        """
        Saves the trained model to a specified path.

        Args:
            model_weight_saving_path (str): The path where the model weights will be saved. If not provided, it defaults to the path specified in the config.

        Returns:
            None
        """
        if model_weight_saving_path:
            self.model.save(model_weight_saving_path)
        else:
            self.model.save(self.config['model_save_path'])

    def load_model(self, model_save_path=None):
            """
            Load the model from the specified path.

            Args:
                model_save_path (str): The path where the model is saved. If not provided, it defaults to config['model_save_path'].
            """
            if model_save_path is None:
                model_save_path = self.config.get('model_save_path')
            
            if model_save_path and os.path.exists(model_save_path):
                print(f"Loading full model from {model_save_path}")
                self.model = tf.keras.models.load_model(model_save_path, custom_objects={
                    'UpdatedMeanIoU': UpdatedMeanIoU,
                })
                self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
            else:
                print(f"Model file not found at {model_save_path}. Starting with a new model.")

    def load_from_checkpoint(self):
        if os.path.exists(self.config['checkpoint_path']):
            print(f"Loading weights from {self.config['checkpoint_path']}")
            self.model.load_weights(self.config['checkpoint_path'])
        else:
            print("Checkpoint not found, starting training from scratch.")

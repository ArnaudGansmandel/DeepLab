# src/training/trainer.py
import datetime
import os
import tensorflow as tf
import keras
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from training.callback import create_callbacks


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # Create callbacks
        self.callbacks = create_callbacks(config)

    def _set_decay_steps(self):
        train_data_size =  tf.data.experimental.cardinality(self.train_dataset).numpy()
        steps_per_epoch = int(train_data_size / self.config['batch_size'])
        num_train_steps = steps_per_epoch * self.config['epochs']

        return num_train_steps
    
    def get_model(self):
        poly_decay = PolynomialDecay(initial_learning_rate=self.config['learning_rate'], decay_steps=self._set_decay_steps(), power=0.9)
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=poly_decay, weight_decay=self.config['weight_decay'])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = [
            'accuracy', 
            tf.keras.metrics.SparseCategoricalAccuracy(), 
            tf.keras.metrics.MeanIoU(num_classes=self.config['num_classes'], sparse_y_pred=False)
        ]
    
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)

    def train(self):
        self.get_model()
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.config['epochs'],
            callbacks=self.callbacks
        )
        return history
    
    def evaluate(self):
        results = self.model.evaluate(self.val_dataset)
        return results

    def load_from_checkpoint(self):
        if os.path.exists(self.config['checkpoint_filepath']):
            print(f"Loading weights from {self.config['checkpoint_filepath']}")
            self.model.load_weights(self.config['checkpoint_filepath'])
        else:
            print("Checkpoint not found, starting training from scratch.")

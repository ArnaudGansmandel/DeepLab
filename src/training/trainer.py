# src/training/trainer.py
import datetime
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from training.learning_rate import PolyDecay
from training.metrics import UpdatedMeanIoU

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        poly_decay = PolyDecay(initial_learning_rate=config['learning_rate'], max_epochs=config['epochs'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=poly_decay, weight_decay=0.0005)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = ['accuracy', tf.keras.metrics.SparseCategoricalAccuracy(), UpdatedMeanIoU(num_classes=config['num_classes'], name='mean_iou')]
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
                
        # Setup logging directory
        log_dir = os.path.join("results/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Setup checkpointing directory
        checkpoint_dir = os.path.dirname(config['checkpoint_path'])
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_callback = ModelCheckpoint(
            filepath=config['checkpoint_path'],
            save_best_only=True,
            monitor='mean_iou',
            mode='max',  
            verbose=1
        )

        # Setup callbacks
        self.callbacks = [
            EarlyStopping(monitor='val_mean_io_u', mode='max', patience=3, restore_best_weights=True),
            ModelCheckpoint(filepath=config['checkpoint_path'], save_best_only=True),
            self.checkpoint_callback,
            self.tensorboard_callback
        ]
    
    def train(self):
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

    def save_model(self, model_weight_saving_path=None):
        if model_weight_saving_path:
            self.model.save(model_weight_saving_path)
        else:
            self.model.save(self.config['model_save_path'])

    def load_from_checkpoint(self):
        if os.path.exists(self.config['checkpoint_path']):
            print(f"Loading weights from {self.config['checkpoint_path']}")
            self.model.load_weights(self.config['checkpoint_path'])
        else:
            print("Checkpoint not found, starting training from scratch.")

# src\training\callback.py
import tensorflow as tf
import os
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


def create_callbacks(config):
    # Setup logging directory for TensorBoard
    log_dir = os.path.join("results/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Setup model checkpointing
    checkpoint_dir = os.path.dirname(config['checkpoint_filepath'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=config['checkpoint_filepath'],
        save_best_only=True,
        save_weights_only=True,
        monitor='val_mean_io_u',
        mode='max',
        verbose=1
    )

    # Setup early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_mean_io_u',
        mode='max',
        patience=3,
        restore_best_weights=True
    )

    # Combine callbacks into a list
    callbacks = [
        early_stopping_callback,
        checkpoint_callback,
        tensorboard_callback
    ]

    return callbacks

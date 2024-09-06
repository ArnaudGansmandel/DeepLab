# scripts\finetune.py
from data_pipeline import data_loader
from deep_lab.models.model import DeepLabV3Plus
from data_pipeline.data_loader import DataLoader
import tensorflow as tf
from training.trainer import Trainer
from utils.plotting import show_predictions

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

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    # Load data
    data_loader = DataLoader(config)

    # Create the datasets
    val_dataset = data_loader.load_data('val', augment=False)
    trainval_dataset = data_loader.load_data('trainval', augment=False)

    ## Fine-tune the model
    # Create the model to be fine-tuned
    model = DeepLabV3Plus(ouput_stride=8, finetuning=True)

    # Restore the model weights from the top model
    model.load_weights('results/models/top_model.weights.h5')

    # Unfreeze the model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it due to the 
    # finetuning argument. This means that the batchnorm layers will 
    # not update their batch statistics. This prevents the batchnorm 
    # layers from undoing all the training we've done so far.    
    model.trainable = True

    # Set the learning rate to be the fine-tuning learning rate
    config['learning_rate']=config['fine_tuning_learning_rate']

    # Train the model
    trainer = Trainer(model=model, train_dataset=trainval_dataset, val_dataset=val_dataset, config=config)

    trainer.train()

    # Save the final model
    trainer.model.save_weights('results/models/fine_tuned_model.weights.h5')
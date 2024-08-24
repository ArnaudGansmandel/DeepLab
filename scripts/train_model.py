from deep_lab.model import DeepLabV3Plus
from data_pipeline.data_loader import DataLoader
import tensorflow as tf
from training.trainer import Trainer

# Training configuration
config = {
    'learning_rate': 0.001,
    'fine_tuning_learning_rate': 0.00001,
    'epochs': 50,
    'batch_size': 16,
    'num_classes': 21,
    'img_size': 224,
    'checkpoint_path': 'results/checkpoints/model.keras',
    'model_save_path': 'results/models/top_model.h5',
}

if __name__ == "__main__":
     # Load data
    data_loader = DataLoader(config)

    # Create the datasets
    train_dataset = data_loader.load_data('train', augment=True)
    val_dataset = data_loader.load_data('val', augment=True)
    trainval_dataset = data_loader.load_data('trainval', augment=True)

    # Create the model
    model = DeepLabV3Plus(dropout_rate=0.2, ouput_stride=16)

    # Create a dummy input tensor with the shape (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, config['img_size'], config['img_size'], 3))

    # Pass the dummy input through the model to initialize the layers
    _ = model(dummy_input)

    # Freeze the ResNet backbone
    model.backbone.resnet_model.trainable = False

    # Create a Trainer instance
    trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, config=config)

    # Load model from checkpoint if available
    trainer.load_from_checkpoint()

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model()

    ## Fine-tune the model

    # Set the learning rate to be the same as the fine-tuning learning rate
    config['learning_rate']=config['fine_tuning_learning_rate']

    # Update the model output stride
    model.update_output_stride(8)

    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.    
    model.backbone.resnet_model.trainable = True

    # Train the model
    trainer = Trainer(model=model, train_dataset=trainval_dataset, val_dataset=val_dataset, config=config)
    trainer.train()

    # Save the final model
    config['model_save_path'] = 'results/models/fine_tuned_model.h5'
    trainer.save_model()
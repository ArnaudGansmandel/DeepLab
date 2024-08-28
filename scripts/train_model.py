from deep_lab.model import DeepLabV3Plus
from data_pipeline.data_loader import DataLoader
import tensorflow as tf
from training.trainer import Trainer

# Training configuration
config = {
    'learning_rate': 0.007,
    'fine_tuning_learning_rate': 0.001,
    'weight_decay': 0.0003,
    'epochs': 5,
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
    train_dataset = data_loader.load_data('train', augment=True)
    val_dataset = data_loader.load_data('val', augment=False)
    trainval_dataset = data_loader.load_data('trainval', augment=False)

    # Create the model
    model = DeepLabV3Plus(ouput_stride=16)

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
    trainer.model.save_weights('results/models/top_model.weights.h5')

    ## Fine-tune the model

    # Set the learning rate to be the fine-tuning learning rate
    config['learning_rate']=config['fine_tuning_learning_rate']

    # Create the model to be fine-tuned
    model = DeepLabV3Plus(ouput_stride=8, finetuning=True)

    # Pass the dummy input through the model to initialize the layers
    _ = model(dummy_input)

    # Unfreeze the model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it due to the 
    # finetuning argument. This means that the batchnorm layers will 
    # not update their batch statistics. This prevents the batchnorm 
    # layers from undoing all the training we've done so far.    
    model.trainable = True

    # Train the model
    trainer = Trainer(model=model, train_dataset=trainval_dataset, val_dataset=val_dataset, config=config)
    trainer.train()

    # Save the final model
    config['model_save_path'] = 'results/models/fine_tuned_model.weights.h5'
    trainer.model.save_weights('results/models/fine_tuned_model.weights.h5')

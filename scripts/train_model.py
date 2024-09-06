# scripts\train_model.py
from deep_lab.models.model import DeepLabV3Plus
from data_pipeline.data_loader import DataLoader
import tensorflow as tf
from training.trainer import Trainer
from utils.plotting import show_predictions

# Training configuration
config = {
    'learning_rate': 0.007,
    'fine_tuning_learning_rate': 0.00001,
    'epochs': 2,
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
    trainaug_dataset = data_loader.load_data('train', augment=True)
    val_dataset = data_loader.load_data('val', augment=False)
    trainval_dataset = data_loader.load_data('trainval', augment=False)

    # Create the model
    model = DeepLabV3Plus(output_stride=16)

    # Create a dummy input tensor with the shape (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, config['img_size'], config['img_size'], 3))

    # Pass the dummy input through the model to initialize the layers
    _ = model(dummy_input)

    # Freeze the ResNet backbone
    # model.backbone.resnet_model.trainable = False
    # model.resnet_model.trainable = False

    for layers in model.backbone.resnet_model.layers:
        layers.trainable = False

    # for layers in model.backbone.backbone.layers:
    #     layers.trainable = False

    # model.backbone.backbone.trainable = False

    # Create a Trainer instance
    trainer = Trainer(model=model, train_dataset=trainaug_dataset, val_dataset=val_dataset, config=config)

    # Load model from checkpoint if available
    #trainer.load_from_checkpoint()

    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.model.save_weights('results/models/top_model.weights.h5')



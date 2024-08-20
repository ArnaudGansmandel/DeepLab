# from data_pipeline.preprocessing.data_preprocessing_old import create_tf_dataset, load_and_preprocess_data
from deep_lab.model import DeepLabV3Plus
from data_pipeline.data_loader import DataLoader
from training.trainer import Trainer

# Training configuration
config = {
    'learning_rate': 0.001,
    'epochs': 20,
    'batch_size': 16,
    'num_classes': 21,
    'img_size': 224,
    'checkpoint_path': 'results/checkpoints/model.keras',
    'model_save_path': 'results/models/model.h5',
}

if __name__ == "__main__":
     # Load data
    data_loader = DataLoader(config)

    # Create the datasets
    train_dataset = data_loader.load_data('train', augment=True)
    val_dataset = data_loader.load_data('val', augment=False)
    trainval_dataset = data_loader.load_data('trainval', augment=True)

    # Create the model
    model = DeepLabV3Plus()

    # Create a Trainer instance
    trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, config=config)
    
    # Load model from checkpoint if available
    trainer.load_from_checkpoint()
    
    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the final model
    trainer.save_model()
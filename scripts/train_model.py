from data_pipeline.preprocessing.data_processing import create_tf_dataset, load_and_preprocess_data
from deep_lab.model import DeepLabV3Plus
# from src.data.data_loader import load_data
from training.trainer import Trainer

# Training configuration
config = {
    'learning_rate': 0.001,
    'epochs': 20,
    'checkpoint_path': 'results/checkpoints/model.keras',
    'model_save_path': 'results/models/model.h5',
    'num_classes': 21
}

def main(train_dataset, val_dataset):
    # Load data
    #train_dataset, val_dataset = load_data('path_to_data')

    # Create the model
    model = DeepLabV3Plus()

    # Create a Trainer instance
    trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, config=config)
    
    # Load model from checkpoint if available
    trainer.load_model()
    
    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the final model
    trainer.save_model()

if __name__ == "__main__":
    # Step 1: Preprocess the data
    image_dir = r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\DeepLab\data\VOCdevkit\VOC2012\SegmentationClass'
    mask_dir = r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\DeepLab\data\VOCdevkit\VOC2012\SegmentationObject'

    train_images, val_images, train_masks, val_masks = load_and_preprocess_data(image_dir, mask_dir)

    # Step 2: Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_images, train_masks)
    val_dataset = create_tf_dataset(val_images, val_masks)

    main(train_dataset, val_dataset)
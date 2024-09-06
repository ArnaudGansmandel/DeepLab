# evaluate.py --  files responsible for the evaluation of different models. Loading and selecting the best model

from deep_lab.models.model import DeepLabV3Plus
from data_pipeline.data_loader import DataLoader
from training.trainer import Trainer

# Training configuration
config = {
    'learning_rate': 0.001,
    'fine_tuning_learning_rate': 0.00001,
    'epochs': 20,
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
    test_dataset = data_loader.load_data('test')

    # Create the model
    model = DeepLabV3Plus()
    model.load_weights('results/models/top_model.weights.h5')

    model.evaluate(test_dataset)
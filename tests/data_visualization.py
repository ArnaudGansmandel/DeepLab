from data_pipeline.data_loader import DataLoader
from deep_lab.models.model import DeepLabV3Plus
import tensorflow as tf
from training.trainer import Trainer
from utils.plotting import show_predictions

# Training configuration
config = {
    'learning_rate': 0.007,
    'fine_tuning_learning_rate': 0.00001,
    'weight_decay': 0.0003,
    'epochs': 5,
    'batch_size': 16,
    'num_classes': 21,
    'img_size': 224,
    'checkpoint_filepath': 'results/checkpoints/model.weights.h5',
    'model_save_path': 'results/models/top_model.weights.h5',
}

data_loader = DataLoader(config)
trainval_dataset = data_loader.load_data('trainval')

deep_lab_model = DeepLabV3Plus(output_stride=16)
deep_lab_model.load_weights('results/models/top_model.weights.h5')

show_predictions(deep_lab_model, dataset=trainval_dataset, num=3)


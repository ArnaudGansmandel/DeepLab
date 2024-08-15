import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from data_pipeline.preprocessing.data_processing import create_tf_dataset, load_and_preprocess_data
from deep_lab.learning_rate import PolyDecay


def train_and_save_model(model_to_train, path_to_save_model=None):
    # Step 1: Preprocess the data
    image_dir = r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\DeepLab\data\VOCdevkit\VOC2012\SegmentationClass'
    mask_dir = r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\DeepLab\data\VOCdevkit\VOC2012\SegmentationObject'

    train_images, val_images, train_masks, val_masks = load_and_preprocess_data(image_dir, mask_dir)

    # Step 2: Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_images, train_masks)
    val_dataset = create_tf_dataset(val_images, val_masks)

    # Step 3: Train the model
    initial_lr = 0.001
    epochs = 50
    poly_decay = PolyDecay(initial_learning_rate=initial_lr, max_epochs=epochs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=poly_decay, weight_decay=0.0005)

    model = model_to_train
    model.compile(optimizer=optimizer, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=21)])

    # Définition de l'arrêt anticipé (early stopping)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=5, callbacks=[early_stopping])

    if path_to_save_model is not None:
        model.save(path_to_save_model)  

if __name__ == "__main__":
    from deep_lab.model import DeepLabV3Plus

    train_and_save_model(DeepLabV3Plus(dropout_rate=0.3), r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\DeepLab\results\models\deep_lab_v3_plus_model.h5')
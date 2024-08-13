
from src.deep_lab.data_processing import create_tf_dataset, load_and_preprocess_data


def train_and_save_model(model_to_train, path_to_save_model):
    # Step 1: Preprocess the data
    image_dir = r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\CodeV3\deep-lab2\data\VOCdevkit\VOC2012\SegmentationClass'
    mask_dir = r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\CodeV3\deep-lab2\data\VOCdevkit\VOC2012\SegmentationObject'

    train_images, val_images, train_masks, val_masks = load_and_preprocess_data(image_dir, mask_dir)

    # Step 2: Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_images, train_masks)
    val_dataset = create_tf_dataset(val_images, val_masks)

    # Step 3: Train the model
    model = model_to_train
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

    model.save(path_to_save_model)

if __name__ == "__main__":
    from src.deep_lab.model_gpt import DeepLabV3Plus

    train_and_save_model(DeepLabV3Plus, r'D:\01_Arnaud\Etudes\04_CNAM\RCP209\Projet\CodeV3\deep-lab2\saved_model\my_model.h5')
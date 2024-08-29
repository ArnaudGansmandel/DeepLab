import tensorflow as tf
from tensorflow.keras.models import Model

from deep_lab.layers import ASPP, Decoder, Backbone
from utils.network_inspection import print_all_layers, print_trainable_variables

class DeepLabV3Plus(Model):
    def __init__(
            self,
            ouput_stride=8,
            finetuning=False,
            name="DeepLabV3Plus",
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.ouput_stride = ouput_stride
        self.finetuning = finetuning
        self.backbone = Backbone()
        self.aspp = ASPP()
        self.decoder = Decoder()

    @tf.function
    def call(self, inputs, training=None):
        if self.finetuning: training = False  # Put the model in inference mode to keep BatchNorm in inference if fine-tuning
        x, low_level_feature = self.backbone(inputs, training=training)
        x = self.aspp(x,  training=training)
        x = self.decoder(x, low_level_feature, training=training)
        return x
    
# Example usage
if __name__ == "__main__":
    IMG_SIZE = 224
    NUM_CLASSES = 21
    input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))    

    print("DeepLabV3Plus summary")
    deeplab_model_training = DeepLabV3Plus( ouput_stride=16)

    # Create a dummy input tensor with the shape (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, IMG_SIZE, IMG_SIZE, 3))

    # Pass the dummy input through the model to initialize the layers
    _ = deeplab_model_training(dummy_input)

    # Now, the model is built, and you can set the ResNet layers to non-trainable
    deeplab_model_training.backbone.resnet_model.trainable = False

    deeplab_model_training.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    deeplab_model_training.summary()

    print_all_layers(deeplab_model_training)
    print_trainable_variables(deeplab_model_training)

import tensorflow as tf
from tensorflow.keras.models import Model

from deep_lab.layers import ASPP, Decoder, Backbone

@tf.keras.utils.register_keras_serializable()
class DeepLabV3Plus(Model):
    def __init__(
            self,
            dropout_rate=0.0,
            name="DeepLabV3Plus",
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.backbone = Backbone(dropout_rate=dropout_rate)
        self.aspp = ASPP(dropout_rate=dropout_rate)
        self.decoder = Decoder(dropout_rate=dropout_rate)

    def call(self, inputs, training=None):
        x, low_level_feature = self.backbone(inputs, training=training)
        x = self.aspp(x, training=training)
        x = self.decoder(x, low_level_feature, training=training)
        return x
    
# Example usage
if __name__ == "__main__":
    IMG_SIZE = 224
    NUM_CLASSES = 21
    input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))    

    print("DeepLabV3Plus summary")
    deeplab_model_training = DeepLabV3Plus()
    output = deeplab_model_training(input)

    deeplab_model_training = Model(inputs=input, outputs=output)
    deeplab_model_training.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    deeplab_model_training.summary()

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
        super(DeepLabV3Plus, self).__init__(name=name, **kwargs)
        self.backbone = Backbone(dropout_rate=dropout_rate)
        self.aspp = ASPP(dropout_rate=dropout_rate)
        self.decoder = Decoder(dropout_rate=dropout_rate)

    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        x = self.aspp(x, training=training)
        low_level_feature = self.backbone.resnet_model.get_layer('conv2_block3_out').output  
        low_level_feature = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x))(low_level_feature)  # Avoid backpropagation
        x = self.decoder(x, low_level_feature, training=training)
        return x

# Example usage
if __name__ == "__main__":
    IMG_SIZE = 224
    NUM_CLASSES = 21
    input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))    

    print("DeepLabV3Plus in training mode")
    deeplab_model_training = DeepLabV3Plus()
    output = deeplab_model_training(input, training=True)

    deeplab_model_training = Model(inputs=input, outputs=output)
    deeplab_model_training.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    deeplab_model_training.summary()

    print("DeepLabV3Plus in inference mode")
    deeplab_model_inference = DeepLabV3Plus()
    output = deeplab_model_inference(input, training=False)

    deeplab_model_inference = Model(inputs=input, outputs=output)
    deeplab_model_inference.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    deeplab_model_inference.summary()

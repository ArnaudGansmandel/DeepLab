import tensorflow as tf
from tensorflow.keras import Model

from deep_lab3.layers import ASPP, Decoder, Backbone
from utils.network_inspection import print_all_layers

@tf.keras.utils.register_keras_serializable()
class DeepLabV3Plus(Model):
    def __init__(
            self,
            num_classes=21,
            dropout_rate=0.0,
            name="DeepLabV3Plus",
            **kwargs
        ):
        #super(DeepLabV3Plus, self).__init__(name=name, **kwargs)
        super().__init__(name=name, **kwargs)
        self.backbone = Backbone(dropout_rate=dropout_rate)
        self.aspp = ASPP(dropout_rate=dropout_rate)
        self.decoder = Decoder(num_classes, dropout_rate=dropout_rate)

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

    model = DeepLabV3Plus(dropout_rate=0.5)
    input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    output = model(input)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Print summary and all layers
    model.summary()
    print_all_layers(model)

# src\deep_lab\model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet101
from tensorflow.keras import layers


from deep_lab.layers.layers import ASPP, Decoder, Backbone
from utils.network_inspection import print_all_layers, print_trainable_variables

class DeepLabV3Plus(Model):
    def __init__(
            self,
            output_stride=8,
            finetuning=False,
            name="DeepLabV3Plus",
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.finetuning = finetuning
        self.backbone = Backbone(output_stride=output_stride)
        self.aspp = ASPP(output_stride=output_stride)
        self.decoder = Decoder(output_stride=output_stride)

        self.final_conv = layers.Conv2D(21, kernel_size=1, kernel_initializer='he_normal', )

    def modify_resnet_layers(self):
        for layer in self.backbone.resnet_model.layers:
            if 'conv5_block' in layer.name:
                if isinstance(layer, layers.Conv2D):
                    if 'conv5_block' in layer.name:
                        layer.strides = (1, 1)
                        layer.dilation_rate = 2
                        
    @tf.function
    def call(self, inputs, training=None):
        if self.finetuning: training = False  # In order to keep BatchNorm in inference during fine-tuning
        self.modify_resnet_layers()
        x, low_level_feature = self.backbone(inputs, training=training)
        x = self.aspp(x, training=training)
        x = layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(x)
        x = self.final_conv(x)
        # x = self.decoder(x, low_level_feature, training=training)
        return x
    
# Example usage
if __name__ == "__main__":
    IMG_SIZE = 224
    NUM_CLASSES = 21
    input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))    

    print("DeepLabV3Plus summary")
    deeplab_model_init = DeepLabV3Plus(output_stride=16)

    # Create a dummy input tensor with the shape (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, IMG_SIZE, IMG_SIZE, 3))

    # Pass the dummy input through the model to initialize the layers
    _ = deeplab_model_init(dummy_input)

    print('without freezing backbone layers')
    print_trainable_variables(deeplab_model_init)

    deeplab_model = DeepLabV3Plus(output_stride=16)
    _ = deeplab_model(dummy_input)
    
    # Now, the model is built, and you can set the ResNet layers to non-trainable
    deeplab_model.backbone.resnet_model.trainable = False
    print('with freezing backbone layers .trainable = False')
    print_trainable_variables(deeplab_model_init)
    print_all_layers(deeplab_model_init)

    # deeplab_model1 = DeepLabV3Plus(output_stride=16)
    # _ = deeplab_model1(dummy_input)
    # deeplab_model1.resnet_model.trainable = False
    # print('with freezing backbone layers .resnet_model.trainable = False')
    # print_trainable_variables(deeplab_model1)


    deeplab_model2 = DeepLabV3Plus(output_stride=16)
    _ = deeplab_model2(dummy_input) 
    for layers in deeplab_model2.backbone.resnet_model.layers:
        layers.trainable = False
    print('with freezing backbone layers .backbone.resnet_model.layers.trainable = False')
    print_trainable_variables(deeplab_model2)

    # deeplab_model_training.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # deeplab_model_training.summary()





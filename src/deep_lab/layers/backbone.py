from deep_lab.layers.base_layers import ConvolutionBlock
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet101


class ResNetBlock(layers.Layer):
    def __init__(
            self, 
            filters=512, 
            dilation_rate=1, 
            multi_grid=(1, 2, 1),
            name='resnet_block',
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.conv1 = ConvolutionBlock(filters, kernel_size=1, dilation_rate=dilation_rate * multi_grid[0])
        self.conv2 = ConvolutionBlock(filters, kernel_size=3, dilation_rate=dilation_rate * multi_grid[1])
        self.conv3 = ConvolutionBlock(filters*4, kernel_size=1, dilation_rate=dilation_rate * multi_grid[2], activation=False)

        self.add = layers.Add()
        self.relu_out = layers.ReLU()
        
    @tf.function
    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)

        x = self.add([x, inputs])
        x = self.relu_out(x)
        return x

class CascadedBlocks(layers.Layer):
    def __init__(
            self,
            num_extra_blocks=3, 
            output_stride=8,
            name='cascaded_blocks',
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.num_extra_blocks = num_extra_blocks
        self.output_stride = output_stride
        self.resnet_blocks = [ResNetBlock() for _ in range(num_extra_blocks)]
    
    def set_dilation_rate(self):
        if self.output_stride == 16:
            dilation_rate = 4
        elif self.output_stride == 8:
            dilation_rate = 8
        else:
            raise ValueError("Unsupported output stride: {}".format(self.output_stride))
        return dilation_rate

    @tf.function
    def call(self, inputs, training=None):
        dilation_rate = self.set_dilation_rate()
        x = inputs
        for i in range(self.num_extra_blocks):
            self.resnet_blocks[i].dilation_rate = (dilation_rate, dilation_rate)
            x = self.resnet_blocks[i](x, training=training)
            dilation_rate *= 2
        return x



# class Backbone(layers.Layer):
#     def __init__(
#             self,
#             output_stride=8,
#             name='backbone',
#             **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.output_stride = output_stride
#         self.resnet_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#         self.cascaded_blocks = CascadedBlocks()
#         self.feature_extractor = FeatureExtractor(self.resnet_model, 'conv2_block3_out')

#     def modify_resnet_layers(self):
#         for layer in self.resnet_model.layers:
#             if 'conv4_block' in layer.name or 'conv5_block' in layer.name:
#                 if isinstance(layer, layers.Conv2D):
#                     if 'conv4_block1_1_conv' in layer.name or 'conv4_block1_0_conv' in layer.name:
#                         layer.strides = (2 if self.output_stride == 16 else 1, 2 if self.output_stride == 16 else 1)
#                     if 'conv4_block' in layer.name:
#                         layer.dilation_rate = (1 if self.output_stride == 16 else 2)
#                     if 'conv5_block' in layer.name:
#                         layer.strides = (1, 1)
#                         layer.dilation_rate = (2 if self.output_stride == 16 else 4)
        
#     @tf.function
#     def call(self, inputs, training=None):
#         self.modify_resnet_layers()
#         # Call resnet_model with training=False in order to keep BatchNorm in inference mode during fine-tuning
#         x = self.resnet_model(inputs, training=False)
#         x = self.cascaded_blocks(x, training=training)

#         low_level_feature = self.feature_extractor(inputs, training=training)
#         low_level_feature = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x))(low_level_feature)  # Avoid backpropagation

#         return x, low_level_feature

class Backbone(layers.Layer):
    def __init__(
            self,
            output_stride=8,
            name='backbone',
            **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_stride = output_stride
        self.resnet_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.feature_extractor = FeatureExtractor(self.resnet_model, 'conv2_block3_out')
        self.feature_extractor.trainable = False
        self.backbone = FeatureExtractor(self.resnet_model, 'conv4_block23_out')
        
    @tf.function
    def call(self, inputs, training=False):
        # Call resnet_model with training=False in order to keep BatchNorm in inference mode during fine-tuning
        # x = self.backbone(inputs, training=training)
        x = self.resnet_model(inputs, training=False)

        low_level_feature = self.feature_extractor(inputs, training=training)
        low_level_feature = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x))(low_level_feature)  # Avoid backpropagation

        return x, low_level_feature
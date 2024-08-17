import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model 
from tensorflow.keras.applications import ResNet101

@tf.keras.utils.register_keras_serializable()
class ConvolutionBlock(layers.Layer):
    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides=1, 
            dilation_rate=1, 
            padding='same', 
            activation=True, 
            dropout_rate=0.0,
            name='convolution_block',
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding, use_bias=False)
        self.bn = layers.BatchNormalization()
        self.activation = activation
        if self.activation:
            self.relu = layers.ReLU()
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)

    @tf.function
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.activation:
            x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=training)
        return x

@tf.keras.utils.register_keras_serializable()
class ResNetBlock(layers.Layer):
    def __init__(
            self, 
            filters=512, 
            dilation_rate=1, 
            multi_grid=(1, 2, 1),
            dropout_rate=0.0,
            name='resnet_block',
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.conv1 = ConvolutionBlock(filters, 1, dilation_rate=dilation_rate * multi_grid[0], activation=True, dropout_rate=dropout_rate)
        self.conv2 = ConvolutionBlock(filters, 3, dilation_rate=dilation_rate * multi_grid[1], activation=True, dropout_rate=dropout_rate)
        self.conv3 = ConvolutionBlock(filters*4, 1, dilation_rate=dilation_rate * multi_grid[2], activation=False, dropout_rate=dropout_rate)

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

@tf.keras.utils.register_keras_serializable()
class CascadedBlocks(layers.Layer):
    def __init__(
            self,
            num_extra_blocks=3, 
            dropout_rate=0.0,
            name='cascaded_blocks',
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.num_extra_blocks = num_extra_blocks
        self.resnet_blocks = [ResNetBlock(dropout_rate=dropout_rate) for _ in range(num_extra_blocks)]
    
    @tf.function
    def call(self, inputs, training=None):
        dilation_rate = 4 if training else 8
        x = inputs
        for i in range(self.num_extra_blocks):
            self.resnet_blocks[i].dilation_rate = (dilation_rate, dilation_rate)
            x = self.resnet_blocks[i](x, training=training)
            dilation_rate *= 2
        return x

@tf.keras.utils.register_keras_serializable()
class FeatureExtractor(layers.Layer):
    def __init__(self, model, layer_name, name='feature_extractor', **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.layer_name = layer_name

    @tf.function
    def call(self, inputs):
        intermediate_model = tf.keras.Model(inputs=self.model.input,
                                            outputs=self.model.get_layer(self.layer_name).output)
        return intermediate_model(inputs)

@tf.keras.utils.register_keras_serializable()
class Backbone(layers.Layer):
    def __init__(
            self,
            dropout_rate=0.0,
            name='backbone',
            **kwargs):
        super().__init__(name=name, **kwargs)
        self.resnet_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.cascaded_blocks = CascadedBlocks(dropout_rate=dropout_rate)
        self.feature_extractor = FeatureExtractor(self.resnet_model, 'conv2_block3_out')

    def modify_resnet_layers(self, output_stride):
        for layer in self.resnet_model.layers:
            if 'conv4_block' in layer.name or 'conv5_block' in layer.name:
                if isinstance(layer, layers.Conv2D):
                    if 'conv4_block1_1_conv' in layer.name or 'conv4_block1_0_conv' in layer.name:
                        layer.strides = (2 if output_stride == 16 else 1, 2 if output_stride == 16 else 1)
                    if 'conv4_block' in layer.name:
                        layer.dilation_rate = (1 if output_stride == 16 else 2)
                    if 'conv5_block' in layer.name:
                        layer.strides = (1, 1)
                        layer.dilation_rate = (2 if output_stride == 16 else 4)
    
    @tf.function
    def call(self, inputs, training=None):
        output_stride = 16 if training else 8
        self.modify_resnet_layers(output_stride=output_stride)
        x = self.resnet_model(inputs, training=training)
        x = self.cascaded_blocks(x, training=training)

        low_level_feature = self.feature_extractor(inputs, training=training)
        low_level_feature = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x))(low_level_feature)  # Avoid backpropagation

        return x, low_level_feature

@tf.keras.utils.register_keras_serializable()
class ASPP(layers.Layer):
    def __init__(
            self, 
            filters=256, 
            dropout_rate=0.0
        ):
        super().__init__()
        self.conv1 = ConvolutionBlock(filters, 1, dropout_rate=dropout_rate)
        self.conv2 = ConvolutionBlock(filters, 3, dropout_rate=dropout_rate)
        self.conv3 = ConvolutionBlock(filters, 3, dropout_rate=dropout_rate)
        self.conv4 = ConvolutionBlock(filters, 3, dropout_rate=dropout_rate)
        self.pool = layers.GlobalAveragePooling2D()
        self.pool_conv = ConvolutionBlock(filters, 1, dropout_rate=dropout_rate)
        self.concat_conv = ConvolutionBlock(filters, 1, dropout_rate=dropout_rate)

    def choice_dilation_rates(self, output_stride):
        if output_stride == 16:
            return [6, 12, 18]
        elif output_stride == 8:
            return [12, 24, 36]
        else:
            raise ValueError("Unsupported output stride: {}".format(output_stride))

    def set_dilation_rates(self, dilation_rates):
        convs = [self.conv2, self.conv3, self.conv4]
        for conv, rate in zip(convs, dilation_rates):
            conv.conv.dilation_rate = (rate, rate)

    @tf.function
    def call(self, inputs, training=None):
        output_stride = 16 if training else 8
        dilation_rates = self.choice_dilation_rates(output_stride)
        self.set_dilation_rates(dilation_rates)

        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)
        x3 = self.conv3(inputs, training=training)
        x4 = self.conv4(inputs, training=training)
        x5 = self.pool(inputs)
        x5 = tf.expand_dims(tf.expand_dims(x5, 1), 1)
        x5 = self.pool_conv(x5, training=training)
        x5 = tf.image.resize(x5, tf.shape(inputs)[1:3])

        x = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        x = self.concat_conv(x, training=training)
        return x

@tf.keras.utils.register_keras_serializable()
class Decoder(layers.Layer):
    def __init__(self, num_classes=21, filters=256, dropout_rate=0.0):
        super().__init__()
        self.decoder_conv1 = ConvolutionBlock(48, 1, dropout_rate=dropout_rate)
        self.decoder_conv2 = ConvolutionBlock(filters, 3, dropout_rate=dropout_rate)
        self.decoder_conv3 = ConvolutionBlock(filters, 3, dropout_rate=dropout_rate)
        self.final_conv = layers.Conv2D(num_classes, 1)

    @tf.function
    def call(self, inputs, low_level_feature, training=None):
        if training:
            x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(inputs)
        else:
            x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(inputs)
        low_level_feature = self.decoder_conv1(low_level_feature, training=training)
        x = tf.concat([x, low_level_feature], axis=-1)
        x = self.decoder_conv2(x, training=training)
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        x = self.decoder_conv3(x, training=training)
        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    # Example usage
    IMG_SIZE = 224
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    input = tf.keras.Input(shape=input_shape)
    resnet_with_atrous_blocks_training = Backbone()
    resnet_with_atrous_blocks_inference = Backbone()

    output_training = resnet_with_atrous_blocks_training(input, training=True)
    output_inference = resnet_with_atrous_blocks_inference(input, training=False)
    
    print("Backbone in training mode")
    resnet_with_atrous_blocks_training = Model(inputs=input, outputs=output_training)
    resnet_with_atrous_blocks_training.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    resnet_with_atrous_blocks_training.summary()

    print("Backbone in inference mode")
    resnet_with_atrous_blocks_inference = Model(inputs=input, outputs=output_inference)
    resnet_with_atrous_blocks_inference.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    resnet_with_atrous_blocks_inference.summary()

import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from tensorflow.keras.applications import ResNet101 # type: ignore

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
        #super(ConvolutionBlock, self).__init__(name=name, **kwargs)
        super().__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding, use_bias=False)
        self.bn = layers.BatchNormalization()
        self.activation = activation
        if self.activation:
            self.relu = layers.ReLU()
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)

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
        #super(ResNetBlock, self).__init__(name=name, **kwargs)
        super().__init__(name=name, **kwargs)
        self.conv1 = ConvolutionBlock(filters, 1, dilation_rate=dilation_rate * multi_grid[0], activation=True, dropout_rate=dropout_rate)
        self.conv2 = ConvolutionBlock(filters, 3, dilation_rate=dilation_rate * multi_grid[1], activation=True, dropout_rate=dropout_rate)
        self.conv3 = ConvolutionBlock(filters*4, 1, dilation_rate=dilation_rate * multi_grid[2], activation=False, dropout_rate=dropout_rate)

        self.add = layers.Add()
        self.relu_out = layers.ReLU()

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
        #super(CascadedBlocks, self).__init__(name=name, **kwargs)
        super().__init__(name=name, **kwargs)
        self.num_extra_blocks = num_extra_blocks
        self.resnet_blocks = [ResNetBlock(dropout_rate=dropout_rate) for _ in range(num_extra_blocks)]
    
    def set_dilation_rates(self, dilation_rates):
        self.resnet_blocks[0].conv1.conv.dilation_rate = (dilation_rates, dilation_rates)
        self.resnet_blocks[0].conv2.conv.dilation_rate = (dilation_rates, dilation_rates)
        self.resnet_blocks[0].conv3.conv.dilation_rate = (dilation_rates, dilation_rates)

        self.resnet_blocks[1].conv1.conv.dilation_rate = (2*dilation_rates, 2*dilation_rates)
        self.resnet_blocks[1].conv2.conv.dilation_rate = (2*dilation_rates, 2*dilation_rates)
        self.resnet_blocks[1].conv3.conv.dilation_rate = (2*dilation_rates, 2*dilation_rates)

        self.resnet_blocks[2].conv1.conv.dilation_rate = (4*dilation_rates, 4*dilation_rates)
        self.resnet_blocks[2].conv2.conv.dilation_rate = (4*dilation_rates, 4*dilation_rates)
        self.resnet_blocks[2].conv3.conv.dilation_rate = (4*dilation_rates, 4*dilation_rates)

    def call(self, inputs, training=None):
        dilation_rate = 4 if training else 8
        x = inputs
        for i in range(self.num_extra_blocks):
            x = self.resnet_blocks[i](x, training=training)
            
            dilation_rate *= 2
        self.set_dilation_rates(dilation_rate)
        return x

@tf.keras.utils.register_keras_serializable()
class Backbone(layers.Layer):
    def __init__(
            self,
            dropout_rate=0.0,
            name='backbone',
            **kwargs):
        #super(Backbone, self).__init__(name=name, **kwargs)
        super().__init__(name=name, **kwargs)
        self.resnet_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.cascaded_blocks = CascadedBlocks(dropout_rate=dropout_rate)

    def modify_resnet_layers(self, output_stride):
        for layer in self.resnet_model.layers:
            if 'conv4_block' in layer.name or 'conv5_block' in layer.name:
                if isinstance(layer, layers.Conv2D):
                    if 'conv4_block1_1_conv' or 'conv4_block1_0_conv' in layer.name:
                        layer.strides = (2 if output_stride == 16 else 1, 2 if output_stride == 16 else 1)
                    if 'conv4_block' in layer.name:
                        layer.dilation_rate = (1 if output_stride == 16 else 2)
                    if 'conv5_block' in layer.name:
                        layer.strides = (1, 1)
                        layer.dilation_rate = (2 if output_stride == 16 else 4)

    def call(self, inputs, training=True):
        output_stride = 16 if training else 8
        self.modify_resnet_layers(output_stride=output_stride)
        x = self.resnet_model(inputs, training=training)
        x = self.cascaded_blocks(x, training=training)
        print(x.shape)
        return x

@tf.keras.utils.register_keras_serializable()
class ASPP(layers.Layer):
    def __init__(
            self, 
            filters=256, 
            dropout_rate=0.0
        ):
        #super(ASPP, self).__init__()
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
            return [1, 6, 12, 18]
        elif output_stride == 8:
            return [1, 12, 24, 36]
        else:
            raise ValueError("Unsupported output stride: {}".format(output_stride))

    def set_dilation_rates(self, dilation_rates):
        self.conv2.conv.dilation_rate = (dilation_rates[1], dilation_rates[1])
        self.conv3.conv.dilation_rate = (dilation_rates[2], dilation_rates[2])
        self.conv4.conv.dilation_rate = (dilation_rates[3], dilation_rates[3])

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
    def __init__(self, num_classes, filters=256, dropout_rate=0.0):
        #super(Decoder, self).__init__()
        super().__init__()
        self.decoder_conv1 = ConvolutionBlock(48, 1, dropout_rate=dropout_rate)
        self.decoder_conv2 = ConvolutionBlock(filters, 3, dropout_rate=dropout_rate)
        self.decoder_conv3 = ConvolutionBlock(filters, 3, dropout_rate=dropout_rate)
        self.final_conv = layers.Conv2D(num_classes, 1, activation='softmax')

    def call(self, inputs, low_level_feature, training=None):
        if training:
            x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(inputs)
            print('training = True')
        else:
            print('training = False')
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
    resnet_with_atrous_blocks = Backbone(dropout_rate=0.5)
    
    input = tf.keras.Input(shape=input_shape)
    output = resnet_with_atrous_blocks(input)
    resnet_with_atrous_blocks = Model(inputs=input, outputs=output)
    resnet_with_atrous_blocks.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    resnet_with_atrous_blocks.summary()
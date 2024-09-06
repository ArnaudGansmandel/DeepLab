from deep_lab.layers.base_layers import ConvolutionBlock
import tensorflow as tf
from tensorflow.keras import layers

class ASPP(layers.Layer):
    def __init__(
            self, 
            filters=256, 
            output_stride=8,
            name='aspp',
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.output_stride = output_stride
        self.conv1 = ConvolutionBlock(filters, kernel_size=1)
        self.conv2 = ConvolutionBlock(filters, kernel_size=3)
        self.conv3 = ConvolutionBlock(filters, kernel_size=3)
        self.conv4 = ConvolutionBlock(filters, kernel_size=3)
        self.pool = layers.GlobalAveragePooling2D()
        self.pool_conv = ConvolutionBlock(filters, kernel_size=1)
        self.concat_conv = ConvolutionBlock(filters, kernel_size=1)

    def choice_dilation_rates(self):
        if self.output_stride == 16:
            return [6, 12, 18]
        elif self.output_stride == 8:
            return [12, 24, 36]
        else:
            raise ValueError("Unsupported output stride: {}".format(self.output_stride))

    def set_dilation_rates(self, dilation_rates):
        convs = [self.conv2, self.conv3, self.conv4]
        for conv, rate in zip(convs, dilation_rates):
            conv.conv.dilation_rate = (rate, rate)
        
    @tf.function
    def call(self, inputs, training=None):
        dilation_rates = self.choice_dilation_rates()
        self.set_dilation_rates(dilation_rates)

        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)
        x3 = self.conv3(inputs, training=training)
        x4 = self.conv4(inputs, training=training)

        x5 = self.pool(inputs)
        x5 = layers.Reshape((1, 1, x5.shape[1]))(x5)
        x5 = self.pool_conv(x5, training=training)
        x5 = layers.UpSampling2D(size=(inputs.shape[1], inputs.shape[2]))(x5)

        x = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        x = self.concat_conv(x, training=training)
        return x
    
class ASPP1(layers.Layer):
    def __init__(
            self, 
            filters=256, 
            output_stride=8,
            name='aspp',
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.output_stride = output_stride
        self.conv1 = ConvolutionBlock(filters, kernel_size=1)
        self.conv2 = ConvolutionBlock(filters, kernel_size=3)
        self.conv3 = ConvolutionBlock(filters, kernel_size=3)
        self.conv4 = ConvolutionBlock(filters, kernel_size=3)
        self.pool = layers.GlobalAveragePooling2D()
        self.pool_conv = ConvolutionBlock(filters, kernel_size=1)
        self.concat_conv = ConvolutionBlock(filters, kernel_size=1)

    def choice_dilation_rates(self):
        if self.output_stride == 16:
            return [6, 12, 18]
        elif self.output_stride == 8:
            return [12, 24, 36]
        else:
            raise ValueError("Unsupported output stride: {}".format(self.output_stride))

    def set_dilation_rates(self, dilation_rates):
        convs = [self.conv2, self.conv3, self.conv4]
        for conv, rate in zip(convs, dilation_rates):
            conv.conv.dilation_rate = (rate, rate)
        
    @tf.function
    def call(self, inputs, training=None):
        dilation_rates = self.choice_dilation_rates()
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
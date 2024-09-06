from deep_lab.layers.base_layers import ConvolutionBlock
import tensorflow as tf
from tensorflow.keras import layers

class Decoder(layers.Layer):
    def __init__(self, 
                 num_classes=21, 
                 filters=256, 
                 output_stride=8,
                 name='decoder',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_stride = output_stride
        self.filters = filters
        self.decoder_conv1 = ConvolutionBlock(48, kernel_size=1)
        self.decoder_conv2 = ConvolutionBlock(filters, kernel_size=3)
        self.decoder_conv3 = ConvolutionBlock(filters, kernel_size=3)
        self.final_conv = layers.Conv2D(num_classes, kernel_size=1, kernel_initializer='he_normal')

    @tf.function
    def call(self, inputs, low_level_feature, training=None):
        if self.output_stride == 16: 
            upsampling = 4
        elif self.output_stride == 8:
            upsampling = 2
        else: 
            raise ValueError("Unsupported output stride: {}".format(self.output_stride))
        x = layers.UpSampling2D(size=(upsampling, upsampling), interpolation='bilinear')(inputs)
        low_level_feature = self.decoder_conv1(low_level_feature, training=training)
        x = tf.concat([x, low_level_feature], axis=-1)
        x = self.decoder_conv2(x, training=training)
        x = self.decoder_conv3(x, training=training)
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        x = self.final_conv(x)
        return x
    
        
class Decoder(layers.Layer):
    def __init__(self, 
                 num_classes=21, 
                 filters=256, 
                 output_stride=8,
                 name='decoder',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_stride = output_stride
        self.filters = filters
        self.decoder_conv1 = ConvolutionBlock(48, kernel_size=1)
        self.decoder_conv2 = ConvolutionBlock(filters, kernel_size=3)
        self.decoder_conv3 = ConvolutionBlock(filters, kernel_size=3)
        self.final_conv = layers.Conv2D(num_classes, kernel_size=1, kernel_initializer='he_normal')        
        
        # Define Conv2DTranspose layers
        if self.output_stride == 16:
            self.up_conv1 = layers.Conv2DTranspose(filters, kernel_size=3, strides=4, padding='same', use_bias=False)
        elif self.output_stride == 8:
            self.up_conv1 = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', use_bias=False)
        else:
            raise ValueError("Unsupported output stride: {}".format(self.output_stride))
        
        self.up_conv2 = layers.Conv2DTranspose(filters, kernel_size=3, strides=4, padding='same', use_bias=False)

    def call(self, inputs, low_level_feature, training=None):
        # Apply first Conv2DTranspose for upsampling
        x = self.up_conv1(inputs)
        
        # Process low-level features
        low_level_feature = self.decoder_conv1(low_level_feature, training=training)
        
        # Concatenate
        x = tf.concat([x, low_level_feature], axis=-1)
        
        # Apply subsequent convolutions
        x = self.decoder_conv2(x, training=training)
        x = self.decoder_conv3(x, training=training)

        # Apply second Conv2DTranspose for further upsampling
        x = self.up_conv2(x)
        
        x = self.final_conv(x)
        return x
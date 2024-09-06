import tensorflow as tf
from tensorflow.keras import layers

class ConvolutionBlock(layers.Layer):
    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides=1, 
            dilation_rate=1, 
            padding='same', 
            activation='elu',  # Default activation set to 'elu'
            kernel_initializer='he_normal',  # Default kernel initializer set to He initialization
            name='convolution_block',
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        
        # Initialize Conv2D layer with custom kernel initializer
        self.conv = layers.Conv2D(
            filters, 
            kernel_size, 
            strides=strides, 
            dilation_rate=dilation_rate, 
            padding=padding, 
            use_bias=False,
            kernel_initializer=kernel_initializer  # Use the specified kernel initializer
        )
        
        # Batch Normalization
        self.bn = layers.BatchNormalization(momentum=0.9997)
        
        # Set the activation function based on input
        self.activation = activation
        if self.activation:
            if activation == 'leaky_relu':
                self.activation_layer = layers.LeakyReLU(alpha=0.1)
            elif activation == 'prelu':
                self.activation_layer = layers.PReLU()
            elif activation == 'elu':
                self.activation_layer = layers.ELU(alpha=1.0)
            else:
                self.activation_layer = layers.Activation(activation)

    @tf.function
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.activation:
            x = self.activation_layer(x)
        return x
    
class ConvolutionBlock_final(layers.Layer):
    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides=1, 
            dilation_rate=1, 
            padding='same', 
            activation=True, 
            name='convolution_block',
            **kwargs
        ):
        super().__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding, use_bias=False)
        self.bn = layers.BatchNormalization(momentum=0.9997)
        self.activation = activation
        if self.activation:
            self.relu = layers.ReLU()

    @tf.function
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.activation:
            x = self.relu(x)
        return x

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
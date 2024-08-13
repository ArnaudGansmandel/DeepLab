import tensorflow as tf
from tensorflow.keras import layers

def print_all_layers(layer, indent=0):
    layer_info = ' ' * indent + f"Layer: {layer.name}, Type: {type(layer).__name__}"

    if hasattr(layer, 'input_shape') and hasattr(layer, 'output_shape'):
        layer_info += f", Input Shape: {layer.input_shape}, Output Shape: {layer.output_shape}"

    if isinstance(layer, layers.Conv2D):
        layer_info += f", Filters: {layer.filters}, Kernel Size: {layer.kernel_size}, Strides: {layer.strides}, Dilation Rate: {layer.dilation_rate}"

    layer_info += f", Input: {layer.input}"
    
    print(layer_info)

    if hasattr(layer, 'layers'):
        for sub_layer in layer.layers:
            print_all_layers(sub_layer, indent + 2)


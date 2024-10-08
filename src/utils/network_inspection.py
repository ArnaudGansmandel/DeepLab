import tensorflow as tf
from tensorflow.keras import layers

def print_all_layers(layer, indent=0):
    layer_info = ' ' * indent + f"Layer: {layer.name}, Type: {type(layer).__name__}"

    if hasattr(layer, 'input_shape') and hasattr(layer, 'output_shape'):
        layer_info += f", Input Shape: {layer.input_shape}, Output Shape: {layer.output_shape}"

    if isinstance(layer, layers.Conv2D):
        layer_info += f", Filters: {layer.filters}, Kernel Size: {layer.kernel_size}, Strides: {layer.strides}, Dilation Rate: {layer.dilation_rate}"
    
    print(layer_info)

    if hasattr(layer, 'layers'):
        for sub_layer in layer.layers:
            print_all_layers(sub_layer, indent + 2)

def print_trainable_variables(model=None, layer=None):
    if hasattr(model, 'layers') and model is not None:
        for layer in model.layers:
                if hasattr(layer, 'layers'):
                    if layer.trainable:
                        print(f"Layer: {layer.name}, Trainable weights: {layer.trainable_weights}\nNon trainable weights: {layer.non_trainable_weights}\n\n")
                elif layer.trainable:
                    print(f"Layer: {layer.name}, Trainable weights: {layer.trainable_weights}\nNon trainable weights: {layer.non_trainable_weights}\n\n")
    elif layer is not None:
        print(f"Layer: {layer.name}, Trainable weights: {layer.trainable_weights}\nNon trainable weights: {layer.non_trainable_weights}\n\n")
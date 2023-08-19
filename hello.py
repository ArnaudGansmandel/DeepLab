import tensorflow as tf 

class Model():
    "Modèle de réseau de neuronnes"

    def __init__(self, nb_layers, epochs=200):
        self.nb_layers = nb_layers
        self.epochs = epochs

    #Modification

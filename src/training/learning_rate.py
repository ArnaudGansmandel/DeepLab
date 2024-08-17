import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class PolyDecay(LearningRateSchedule):
    def __init__(self, initial_learning_rate, max_epochs, power=0.9):
        super(PolyDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.max_epochs = max_epochs
        self.power = power

    def __call__(self, step):
        return self.initial_learning_rate * (1 - step / self.max_epochs)**self.power

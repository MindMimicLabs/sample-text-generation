import yaml
import pathlib
import tensorflow as tf
from typeguard import typechecked

@typechecked
def create_dnn(max_sentence_length: int, unique_vocab: int) -> tf.keras.Sequential:

    HERE = pathlib.Path(__file__).parent
    with open(HERE.joinpath('./dnn_model.yml')) as file:
        FLAGS = yaml.load(file, Loader = yaml.SafeLoader)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            units = FLAGS['units_lstm'],
            input_shape = (max_sentence_length, unique_vocab),
            kernel_initializer = "VarianceScaling",
            kernel_regularizer = tf.keras.regularizers.L1L2(l1 = FLAGS['reg'], l2 = FLAGS['reg'])),
        tf.keras.layers.Dense(unique_vocab),
        tf.keras.layers.Activation("softmax")])

    model.compile(
        optimizer = tf.keras.optimizers.Nadam(lr = FLAGS['lr']),
        loss = 'categorical_crossentropy')

    return model
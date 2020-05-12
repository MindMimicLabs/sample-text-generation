import utils as u
import tensorflow as tf

# Get configuration information --------------------------------------------------
token_count = u.count_total_tokens('../data')

# Make the model -----------------------------------------------------------------
config = u.load_config('./text_generation.yml')
config = config['create']

# the `+1` is to allow for EOS to be added in later
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        units = config['units'],
        input_shape = (config['sequence_length'], token_count + 1),
        kernel_initializer = "VarianceScaling",
        kernel_regularizer = tf.keras.regularizers.L1L2(l1 = config['reg'], l2 = config['reg'])),
    tf.keras.layers.Dense(token_count + 1),
    tf.keras.layers.Activation("softmax")])

model.compile(
    optimizer = tf.keras.optimizers.Nadam(lr = config['lr']),
    loss = 'categorical_crossentropy')

# Save the model -----------------------------------------------------------------
model.save('./model/current.py.model')
model.summary()

import numpy as np
import utils as u
import tensorflow as tf

def create_model():
    # Get configuration information --------------------------------------------------
    config = u.load_config()['create']
    data_path = u.get_data_path()
    token_count = len(np.load(data_path.joinpath('./_token_to_int.npy'), allow_pickle = True).item())    

    # Make the model -----------------------------------------------------------------
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            units = config['units'],
            input_shape = (config['sequence_length'], token_count),
            kernel_initializer = "VarianceScaling",
            kernel_regularizer = tf.keras.regularizers.L1L2(l1 = config['reg'], l2 = config['reg'])),
        tf.keras.layers.Dense(token_count),
        tf.keras.layers.Activation("softmax")])

    model.compile(
        optimizer = tf.keras.optimizers.Nadam(lr = config['lr']),
        loss = 'categorical_crossentropy')
    
    return model

if __name__ == '__main__':
    model = create_model()
    model_path = u.get_model_path()
    model.save_weights(str(model_path))

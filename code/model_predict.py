import random as r
import numpy as np
import tensorflow as tf
import utils as u

# Get configuration information --------------------------------------------------
config = u.load_config()
data_path = u.get_data_path()
token_to_int = np.load(data_path.joinpath('./_token_to_int.npy'), allow_pickle = True).item()
token_count = len(token_to_int)
sequence_length = config['create']['sequence_length']
tokens_to_generate = config['predict']['tokens_to_generate']

# Load the model -----------------------------------------------------------------
model = tf.keras.models.load_model(data_path.joinpath('./current.py.model'))
model.summary()

# Setup `dict` to un-vectorize ---------------------------------------------------
int_to_token = dict((x[1], x[0]) for x in token_to_int.items())

# Seed the prediction process ----------------------------------------------------
# pin our seed for reproduceability
r.seed(0)
for file in data_path.iterdir():
    if file.suffix == '.txt':
        tokens = u.tokenize_document(file)
        vector = u.vectorize(tokens, token_to_int)
        indx = r.randint(0, len(tokens) - sequence_length)
        seed = [x for x in vector[indx:(indx + sequence_length)]]
        break
print(f'Random Seed: {u.vec_to_str(seed, int_to_token)}')

# Generation  --------------------------------------------------------------------
generated_tokens = []
for i in range(tokens_to_generate):
    one_hot_x = u.one_hot_single(seed, token_count)
    one_hot_y = model.predict(one_hot_x)
    y = int(u.un_one_hot(one_hot_y)[0])
    seed = seed[1:sequence_length]
    seed.append(y)
    generated_tokens.append(y)

print(f'Generated Text: {u.vec_to_str(generated_tokens, int_to_token)}')

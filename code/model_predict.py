import create_model as cm
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
model_path = str(u.get_model_path())

# Load the model weights ---------------------------------------------------------
model = cm.create_model()
model.load_weights(model_path)

# Setup `dict` to un-vectorize ---------------------------------------------------
int_to_token = dict((x[1], x[0]) for x in token_to_int.items())

# Seed the prediction process ----------------------------------------------------
# pin our seed for reproduceability
np.random.seed(0)
for file in data_path.iterdir():
    if file.suffix == '.txt':
        tokens = u.tokenize_document(file)
        vector = u.vectorize(tokens, token_to_int)
        indx = np.random.randint(len(tokens) - sequence_length)
        seed = [int(x) for x in vector[indx:(indx + sequence_length)]]
        break
print(f'Seed: {u.vec_to_str(seed, int_to_token)}')

# Generation  --------------------------------------------------------------------
for diversity in [1, 1.33, 1.66, 2]:
    np.random.seed(0)
    curr = seed
    generated_tokens = []
    for i in range(tokens_to_generate):
        one_hot_x = u.one_hot_single(curr, token_count)
        one_hot_y = model.predict(one_hot_x)
        y = u.un_one_hot(one_hot_y, diversity)
        curr = curr[1:sequence_length]
        curr.append(y)
        generated_tokens.append(y)
    print(f'Diversity {diversity} generated Text: {u.vec_to_str(generated_tokens, int_to_token)}')

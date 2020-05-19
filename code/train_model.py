import create_model as cm
import math as m
import numpy as np
import progressbar as pb
import tensorflow as tf
import utils as u

# Get configuration information --------------------------------------------------
config = u.load_config()
data_path = u.get_data_path()
token_count = len(np.load(data_path.joinpath('./_token_to_int.npy'), allow_pickle = True).item())
model_path = str(u.get_model_path())

# Load the model weights ---------------------------------------------------------
model = cm.create_model()
model.load_weights(model_path)

# Training -----------------------------------------------------------------------
for epoch_i in range(1, config['train']['epochs'] + 1):
    data_path_i = 0
    for file in data_path.iterdir():
        if file.suffix == '.npy' and not file.stem.startswith('_'):
            data_path_i = data_path_i + 1
            tokens = np.load(file)
            # the +1 is so our sample is both the x (sequence_length) and y(+1) as a single row
            # `u.make_batch()` splits them up later
            samples = u.make_samples(tokens, config['create']['sequence_length'] + 1)
            sz = samples.shape
            widgets = [ f'Epoch: {epoch_i} Doc: {data_path_i} Batch: ', pb.Counter(format = '%(value)d/%(max_value)d'), ' ', pb.Bar(marker = '.', left = '[', right = ']'), ' ', pb.ETA() ]
            with pb.ProgressBar(widgets = widgets, max_value = m.ceil(sz[0]/config['train']['batch_size'])) as bar:
                batch_i = 0        
                while batch_i < sz[0]:
                    bar.update(batch_i/config['train']['batch_size'])
                    batch = u.make_batch(samples, batch_i, config['train']['batch_size'])
                    one_hot = u.one_hot_batch(batch, token_count)
                    # The `train_on_batch()` function is the most granular training.
                    # In our lab, we like that level of control.
                    # In your case just using `fit()` on the whole thing may make sense
                    loss = model.train_on_batch(one_hot.x, one_hot.y)
                    batch_i = batch_i + config['train']['batch_size']
    print(f'Epoch: {epoch_i}, Loss: {loss}')

# Save the model weights ---------------------------------------------------------
model.save_weights(model_path)

import math as m
import pathlib
import progressbar as pb
import yaml
from typeguard import typechecked

# Bring in any parameters we need  -----------------------------------------------
HERE = pathlib.Path(__file__).parent
with open(HERE.joinpath('./generate_text.yml')) as file:
    FLAGS = yaml.load(file, Loader = yaml.SafeLoader)

# Bring helper functions  --------------------------------------------------------
import load_corpus as lc
import dnn_model as dnn
import utils as u

# Load our corpus and create our model -------------------------------------------
corpus = lc.load_corpus('../data')
# override the coupus length because they are crazy
corpus.max_sentence_length = FLAGS['max_sentence_length']
model = dnn.create_dnn(corpus.max_sentence_length, len(corpus.vocab))

# Training -----------------------------------------------------------------------
for document in corpus.documents:

    samples = u.make_samples(document, corpus.max_sentence_length + 1)
    sz = samples.shape

    # The `train_on_batch()` function is the most granular training.
    # In our lab, we like that level of control.
    # In your case just using `fit()` on the whole thing may make sense
    for i in range(0, FLAGS['epochs']):
        widgets = [ f'Epoch {i+1} batch ', pb.Counter(format = '%(value)d/%(max_value)d'), ' ', pb.Bar(marker = '.', left = '[', right = ']'), ' ', pb.ETA() ]
        with pb.ProgressBar(widgets = widgets, max_value = m.ceil(sz[0]/FLAGS['batch_size'])) as bar:
            batch_i = 0        
            while batch_i < sz[0]:
                bar.update(batch_i/FLAGS['batch_size'])
                batch = u.make_batch(samples, batch_i, FLAGS['batch_size'])
                one_hot = u.one_hot_batch(batch, corpus.vocab)
                loss = model.train_on_batch(one_hot.x, one_hot.y)
                batch_i = batch_i + FLAGS['batch_size']
        print(f'Epoch: {i+1}, Loss: {loss}')



import numpy as np
from collections import namedtuple
from typeguard import typechecked

xy_data = namedtuple('xy_data', 'x, y')

@typechecked
def make_samples(tokens: list, sample_length: int) -> np.array:
    max_word_length = max([len(x) for x in tokens])
    samples = np.empty([len(tokens) - sample_length + 1, sample_length], dtype = f'<U{max_word_length}')
    for i in range(0, samples.shape[0]):
        for j in range(0, samples.shape[1]):
            samples[i,j] = tokens[i + j]
    return samples

@typechecked
def make_batch(samples: np.array, start: int, batch_size: int) -> np.array:
    sz = samples.shape
    batch = samples[range(start, min(sz[0], start + batch_size))]
    batch_x = batch[:, range(0, sz[1]-1)]
    batch_y = batch[:, sz[1]-1]
    return xy_data(batch_x, batch_y)

@typechecked
def one_hot_batch(batch: xy_data, unique_tokens: dict) -> np.array:
    sz = batch.x.shape
    one_hot_x = np.zeros([sz[0], sz[1], len(unique_tokens)], dtype = 'float32')
    one_hot_y = np.zeros([len(batch.y), len(unique_tokens)], dtype = 'float32')
    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            one_hot_x[i, j, unique_tokens[batch.x[i, j]]] = True
    for i in range(0, len(batch.y)):
        one_hot_y[i, unique_tokens[batch.y[i]]] = True
    return xy_data(one_hot_x, one_hot_y)

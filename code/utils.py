import pathlib
import string
import yaml
import numpy as np
from collections import namedtuple
from typeguard import typechecked

xy_data = namedtuple('xy_data', 'x, y')

@typechecked
def make_samples(tokens: np.array, sample_length: int) -> np.array:
    samples = np.empty([len(tokens) - sample_length + 1, sample_length], dtype = int)
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
def one_hot_batch(batch: xy_data, unique_token_count: int) -> np.array:
    sz = batch.x.shape
    one_hot_x = np.zeros([sz[0], sz[1], unique_token_count], dtype = 'float32')
    one_hot_y = np.zeros([len(batch.y), unique_token_count], dtype = 'float32')
    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            one_hot_x[i, j, batch.x[i, j]] = True
    for i in range(0, len(batch.y)):
        one_hot_y[i, batch.y[i]] = True
    return xy_data(one_hot_x, one_hot_y)

@typechecked
def one_hot_single(sample: list, unique_token_count: int) -> np.array:
    one_hot = np.zeros([1, len(sample), unique_token_count], dtype = 'float32')
    for i in range(0, len(sample)):
        one_hot[0, i, sample[i]] = True
    return one_hot

@typechecked
def un_one_hot(sample: np.array, int_to_word: dict) -> list:
    resut = []
    for i in range(0, sample.shape[0]):
        indx = np.argmax(sample[i])
        resut.append(int_to_word[indx])
    return resut

@typechecked
def load_config() -> dict:
    config_path = _resolve_relitive_path('./text_generation.yml')
    with open(config_path) as file:
        return yaml.load(file, Loader = yaml.SafeLoader)

@typechecked
def get_data_path() -> pathlib.Path:
    data_path = _resolve_relitive_path('../data')
    return data_path

@typechecked
def _resolve_relitive_path(relitive_path: str) -> pathlib.Path:
    relitive_path = pathlib.Path(__file__).parent.joinpath(relitive_path).resolve()
    return relitive_path

import pathlib
import yaml
import math as m
import numpy as np
import string as s
from collections import namedtuple
from typeguard import typechecked

xy_data = namedtuple('xy_data', 'x, y')

@typechecked
def tokenize_document(path: pathlib.Path) -> list:
    with open(path, encoding = 'utf8') as document:
        lines = document.readlines()
    lines = tokenize_lines(lines)
    tokens = [x for y in lines for x in y]
    return tokens

@typechecked
def tokenize_lines(lines: list) -> list:
    lines = [tokenize_line(line) for line in lines]
    lines = [line for line in lines if line != None]
    return lines

@typechecked
def tokenize_line(line: str) -> list:
    tokens = line.strip().split()
    tokens = [token.strip().lower() for token in tokens]
    tokens = [token for token in tokens if token != '' and token not in s.punctuation]
    if len(tokens) == 0:
        return None
    return tokens

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
def un_one_hot(one_hot: np.array, temperature: float = 1) -> np.array:
    weights = one_hot.copy().flatten()
    for i in range(0, weights.shape[0]):
        weights[i] = m.exp(m.log(weights[i])/temperature)
    s = np.sum(weights)
    for i in range(0, weights.shape[0]):
        weights[i] = weights[i]/s
    result = np.random.choice(weights.shape[0], p = weights)
    return result

@typechecked
def load_config() -> dict:
    config_path = _resolve_relitive_path('./text_generation.yml')
    with open(config_path) as file:
        return yaml.load(file, Loader = yaml.SafeLoader)

@typechecked
def get_model_path() -> pathlib.Path:
    model_path = get_data_path().joinpath('./model.weights/weights')
    return model_path        

@typechecked
def get_data_path() -> pathlib.Path:
    data_path = _resolve_relitive_path('../data')
    return data_path

@typechecked
def _resolve_relitive_path(relitive_path: str) -> pathlib.Path:
    relitive_path = pathlib.Path(__file__).parent.joinpath(relitive_path).resolve()
    return relitive_path

@typechecked
def vectorize(tokens: list, token_to_int: dict) -> np.array:
    vector = np.zeros(len(tokens), dtype = int)
    for i in range(0, len(tokens)):
        vector[i] = token_to_int[tokens[i]]
    return vector

@typechecked
def unvectorize(vector: np.array, int_to_token: dict) -> list:
    tokens = [int_to_token[x] for x in vector]
    return tokens
    
@typechecked
def vec_to_str(vector: np.array, int_to_token: dict) -> str:
    tokens = unvectorize(vector, int_to_token)
    t_str = ' '.join(str(x) for x in tokens)
    v_str = ' '.join(str(x) for x in vector)
    result = f'{t_str} ({v_str})'
    return result

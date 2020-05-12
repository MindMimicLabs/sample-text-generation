import pathlib
import string
import numpy as np
import progressbar as pb
from typeguard import typechecked

# get the real path to the corpus based on the relative one
corpus_path = pathlib.Path(__file__).parent.joinpath('../data').resolve()
corpus_path_i = 0

# collect all the unique tokens
# EOS stands for 'End of Sequence' and is added in dynamically to the vectorized data
EOS = 'EOS'
unique_tokens = dict({EOS: 0})
unique_tokens_i = 1

@typechecked
def tokenize_document(path: pathlib.Path) -> list:
    with open(file, encoding = 'utf8') as document:
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
    tokens = [token for token in tokens if token != '' and token not in string.punctuation]
    if len(tokens) == 0:
        return None
    tokens.append(EOS)
    return tokens

# we perform this process in two passes: one to collect all the tokens from the _entire_ corpus and another to vectorise each document
widgets = ['Counting: File# ', pb.Counter(), ' ', pb.BouncingBar(marker = '.', left = '[', right = ']'), ' ', pb.Timer()]
with pb.ProgressBar(widgets = widgets) as bar:
    for file in corpus_path.iterdir():
        if file.suffix == '.txt':
            corpus_path_i = corpus_path_i + 1
            bar.update(corpus_path_i)
            tokens = tokenize_document(file)
            for token in tokens:
                if token not in unique_tokens:
                    unique_tokens[token] = unique_tokens_i
                    unique_tokens_i = unique_tokens_i + 1

# save the unique tokens
np.save(corpus_path.joinpath('./_unique_tokens.npy'), unique_tokens)

# vectorise each file
vectorise_i = 0
widgets = ['Vectorising: File# ', pb.Counter(), ' ', pb.Bar(marker = '.', left = '[', right = ']'), ' ', pb.Percentage(), ' ', pb.ETA() ]
with pb.ProgressBar(widgets = widgets, max_value = corpus_path_i) as bar:
    for file in corpus_path.iterdir():
        if file.suffix == '.txt':
            vectorise_i = vectorise_i + 1
            bar.update(vectorise_i)
            tokens = tokenize_document(file)
            document_v = np.empty(len(tokens), dtype = int)
            for i in range(0, len(tokens)):
                document_v[i] = unique_tokens[tokens[i]]            
            np.save(corpus_path.joinpath(f'{file.stem}.npy'), document_v)            

print(f'unique tokens: {unique_tokens_i}')
print(f'corpus documents: {corpus_path_i}')

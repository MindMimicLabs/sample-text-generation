import numpy as np
import progressbar as pb
import utils as u

# get the real path to the corpus based on the relative one
data_path = u.get_data_path()
data_path_i = 0

# collect all the unique tokens
token_to_int = dict()
token_to_int_i = 0

# we perform this process in two passes: one to collect all the tokens from the _entire_ corpus and another to vectorise each document
widgets = ['Counting: File# ', pb.Counter(), ' ', pb.BouncingBar(marker = '.', left = '[', right = ']'), ' ', pb.Timer()]
with pb.ProgressBar(widgets = widgets) as bar:
    for file in data_path.iterdir():
        if file.suffix == '.txt' and not file.stem.startswith('_'):
            data_path_i = data_path_i + 1
            bar.update(data_path_i)
            tokens = u.tokenize_document(file)
            for token in tokens:
                if token not in token_to_int:
                    token_to_int[token] = token_to_int_i
                    token_to_int_i = token_to_int_i + 1

# save the unique tokens
np.save(data_path.joinpath('./_token_to_int.npy'), token_to_int)

# vectorise each file
vectorise_i = 0
widgets = ['Vectorising: File# ', pb.Counter(), ' ', pb.Bar(marker = '.', left = '[', right = ']'), ' ', pb.Percentage(), ' ', pb.ETA() ]
with pb.ProgressBar(widgets = widgets, max_value = data_path_i) as bar:
    for file in data_path.iterdir():
        if file.suffix == '.txt' and not file.stem.startswith('_'):
            vectorise_i = vectorise_i + 1
            bar.update(vectorise_i)
            tokens = u.tokenize_document(file)
            vector = u.vectorize(tokens, token_to_int)
            np.save(data_path.joinpath(f'{file.stem}.npy'), vector)

print(f'unique tokens: {token_to_int_i}')
print(f'corpus documents: {data_path_i}')

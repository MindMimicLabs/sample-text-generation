import pathlib
import string
from typeguard import typechecked

EOS = 'EOS'

class corpus:
    @typechecked
    def __init__(self, documents:list, vocab:dict, max_sentence_length:int) -> None:
        self.documents = documents
        self.vocab = vocab
        self.max_sentence_length = max_sentence_length

@typechecked
def load_corpus(corpus_path: str) -> corpus:    
    HERE = pathlib.Path(__file__).parent
    corpus_path = HERE.joinpath(corpus_path)
    # load the documents
    documents = [x for x in corpus_path.iterdir() if x.suffix == '.txt']
    words_in_documents = []
    vocab = set()
    max_sentence_length = 0
    for i in range(0, len(documents)):
        document = documents[i]
        with open(document, encoding = 'utf8') as document:
            lines = document.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split() for line in lines if line != '']
        lines = [[word.lower() for word in line if word not in string.punctuation] for line in lines]
        # EOS marks the end of a sentence for when we flatten the list later
        for line in lines: line.append(EOS)
        t1 = max([len(line) for line in lines])
        max_sentence_length = max(max_sentence_length, t1)
        words_in_document = [x for y in lines for x in y]
        print(f'document # {i+1}: words: {len(words_in_document)}. max sentence length: {t1}')
        vocab.update(words_in_document)
        words_in_documents.append(words_in_document)
    vocab = dict((v, i) for i, v in enumerate(vocab))
    print(f'unique words: {len(vocab)}')
    return corpus(words_in_documents, vocab, max_sentence_length)

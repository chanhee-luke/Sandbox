"""Module defining input.

Input implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""

from sandbox.input.inputter import collect_feature_vocabs, make_features, \
    collect_features, get_num_features, \
    load_fields_from_vocab, get_fields, \
    save_fields_to_vocab, build_dataset, \
    build_vocab, merge_vocabs, OrderedIterator
from sandbox.input.dataset_base import DatasetBase, PAD_WORD, BOS_WORD, \
    EOS_WORD, UNK
from sandbox.input.text_dataset import TextDataset, ShardedTextCorpusIterator

__all__ = ['PAD_WORD', 'BOS_WORD', 'EOS_WORD', 'UNK', 'DatasetBase',
           'collect_feature_vocabs', 'make_features',
           'collect_features', 'get_num_features',
           'load_fields_from_vocab', 'get_fields',
           'save_fields_to_vocab', 'build_dataset',
           'build_vocab', 'merge_vocabs', 'OrderedIterator',
           'TextDataset',
           'ShardedTextCorpusIterator']

__version__ = '0.0.1'
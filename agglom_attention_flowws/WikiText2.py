"""
This module uses lightly modified code from the keras-transformer
project, made available under the MIT license reproduced below.

The MIT License

Copyright 2018 Kirill Mavreshko (https://www.linkedin.com/in/kirill-mavreshko/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from itertools import islice
from typing import Iterable, List, Optional

import flowws
from flowws import Argument as Arg
from .kt_examples import wikitext
from .kt_examples.bpe import BPEEncoder, ID_FOR_PADDING
import numpy as np

def pad_lm_samples(samples: Iterable[List[int]],
                   required_sequence_length: int):
    tail_padding = [ID_FOR_PADDING]
    for sample in samples:
        assert len(sample) > 0
        sample.extend(tail_padding * (required_sequence_length - len(sample)))

def training_data_to_samples(training_set_name: str,
                             encoder: BPEEncoder,
                             max_sequence_length: int) -> np.ndarray:
    """
    Reads WikiText dataset, interpreting each line as an independent sequence,
    then splits those lines with BPE tokenizer and turns them into word ids
    based on previously constructed BPE vocabulary (both the tokenizer
    and the vocabulary are parts of the BPEEncoder instance).

    Those word id's then packed into a matrix the size of
    (number of lines x max_sequence_length + 1), which can be later sliced
    to get X and Y matrices of sequences for training).
    """
    training_set = wikitext.read_wikitext_file(training_set_name)
    useful_sequences = []
    for line in training_set.splitlines():
        clean_line = line.strip()
        is_header = clean_line.startswith('=') and clean_line.endswith('=')
        if is_header or not clean_line:
            continue
        # the encoder is supposed to add <SEQ> and </SEQ>
        id_word_pairs = list(encoder(clean_line))
        useful_sequences.append(
            [word_id for word_id, _ in id_word_pairs[:max_sequence_length]])

    pad_lm_samples(useful_sequences, max_sequence_length + 1)
    result = np.empty(
        (len(useful_sequences), max_sequence_length + 1),
        dtype='int32')
    for i, sequence in enumerate(useful_sequences):
        result[i, :] = sequence
    return result

def training_data_to_dense_samples(training_set_name: str,
                                   encoder: BPEEncoder,
                                   max_sequence_length: int) -> np.ndarray:
    """
    Reads WikiText dataset, interpreting each line as an independent sequence,
    then splits those lines with BPE tokenizer and turns them into word ids
    based on previously constructed BPE vocabulary (both the tokenizer
    and the vocabulary are parts of the BPEEncoder instance).

    Those word id's then packed into a matrix the size of
    (number of lines x max_sequence_length + 1), which can be later sliced
    to get X and Y matrices of sequences for training).
    """
    training_set = wikitext.read_wikitext_file(training_set_name)
    useful_sequences = []

    def stream_bpe_tokens():
        for line in training_set.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            # the encoder is supposed to add <SEQ> and </SEQ>
            id_word_pairs = encoder(clean_line)
            yield from id_word_pairs

    id_word_stream = stream_bpe_tokens()
    while True:
        chunk = list(islice(id_word_stream, max_sequence_length))
        if len(chunk) == 0:
            break
        sample_sequence = [word_id for word_id, _ in chunk]
        useful_sequences.append(sample_sequence)

    pad_lm_samples(useful_sequences, max_sequence_length + 1)
    result = np.empty(
        (len(useful_sequences), max_sequence_length + 1),
        dtype='int32')
    for i, sequence in enumerate(useful_sequences):
        result[i, :] = sequence
    return result

@flowws.add_stage_arguments
class WikiText2(flowws.Stage):
    """Train a model on the wikitext-2 dataset"""

    ARGS = [
        Arg('sequence_length', '-l', int, 64,
            help='Maximum sequence length of the network'),
        Arg('batch_size', '-b', int, 32,
            help='Batch size for training')
        ]

    def run(self, scope, storage):
        encoder = wikitext.build_wikitext_bpe_encoder()
        scope['vocabulary_size'] = encoder.vocabulary_size()
        sequence_length = self.arguments['sequence_length']

        def x_y_for_dataset(dataset_name):
            fat_sample = training_data_to_dense_samples(
                dataset_name, encoder, sequence_length)
            _x = fat_sample[:, :sequence_length]
            _y = np.expand_dims(fat_sample[:, 1:], axis=-1)
            return _x, _y

        scope['training_data'] = x_y_for_dataset(wikitext.TRAINING_SET_NAME)
        (scope['x_train'], scope['y_train']) = scope['training_data']
        scope['validation_data'] = x_y_for_dataset(wikitext.VALIDATION_SET_NAME)
        scope['test_data'] = x_y_for_dataset(wikitext.TEST_SET_NAME)
        scope['loss'] = 'sparse_categorical_crossentropy'
        scope['sequence_length'] = sequence_length
        scope['batch_size'] = self.arguments['batch_size']
        scope['encoder'] = lambda x: [n for (n, token) in encoder(x)]
        scope['decoder'] = lambda x: encoder.decode(x)

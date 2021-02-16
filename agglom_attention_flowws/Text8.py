import functools
import os

import flowws
from flowws import Argument as Arg
import numpy as np

TEXT8_LOCATION = os.path.join(os.path.dirname(__file__), 'text8.npz')

class Text8DataWrapper:
    def __init__(self):
        self.text8_data = np.load(TEXT8_LOCATION)['text8']

        # remove spaces in the ASCII character set, given that we have a-z and ' '
        # reserve index 0 for masking later, if desired
        compressed_characters = [0]

        compressed_characters.extend(range(ord('a'), ord('z') + 1))
        compressed_characters.append(ord(' '))

        self.compressed_characters = np.array(
            compressed_characters, dtype=np.uint8)

        self.compress_character_map = np.zeros(256, dtype=np.uint8)
        for (i, j) in enumerate(self.compressed_characters):
            self.compress_character_map[j] = i

        self.vocabulary_size = len(self.compressed_characters)
        self.char_map = {i: chr(c) for (i, c) in enumerate(self.compressed_characters)}
        self.inv_char_map = {chr(c): i for (i, c) in enumerate(self.compressed_characters)}

    def random_batch(self, batch_size, seq_len, use_fractions=(0, 1.)):
        N = len(self.text8_data)
        whole_start_index = int(N*use_fractions[0])
        whole_end_index = int(N*use_fractions[1])

        while True:
            start_indices = np.random.randint(
                whole_start_index, whole_end_index - seq_len,
                size=batch_size)
            end_indices = start_indices + seq_len + 1

            slices = np.array(
                [self.compress_character_map[self.text8_data[start:end]]
                 for (start, end) in zip(start_indices, end_indices)],
                dtype=self.text8_data.dtype)

            inputs = slices[:, :-1]
            outputs = slices[:, 1:, np.newaxis]
            yield (inputs, outputs)

    def encode(self, text):
        return np.array([self.inv_char_map[c] for c in text])

    def decode(self, numbers):
        return ''.join(self.char_map[x] for x in numbers)

@flowws.add_stage_arguments
class Text8(flowws.Stage):
    """Train a model on the text8 dataset"""

    ARGS = [
        Arg('sequence_length', '-l', int, 64,
            help='Maximum sequence length of the network'),
        Arg('validation_fraction', None, float, .3,
            help='Fraction of the source file to use as validation data'),
        Arg('test_fraction', None, float, .1,
            help='Fraction of the source file to use as test data'),
        Arg('epoch_scaling_factor', None, float, 1,
            help='Fraction of training data to use per epoch'),
        Arg('batch_size', '-b', int, 32,
            help='Batch size for training')
        ]

    def run(self, scope, storage):
        dataset = Text8DataWrapper()
        scope['vocabulary_size'] = dataset.vocabulary_size

        sequence_length = self.arguments['sequence_length']
        batch_size = self.arguments['batch_size']
        validation_fraction = self.arguments['validation_fraction']
        test_fraction = self.arguments['test_fraction']
        epoch_scaling_factor = self.arguments['epoch_scaling_factor']

        fractions = 1 - np.cumsum([test_fraction, validation_fraction])[::-1]
        train_data = dataset.random_batch(
            batch_size, sequence_length, (0, fractions[0]))
        val_data = dataset.random_batch(
            batch_size, sequence_length, (fractions[0], fractions[1]))
        test_data = dataset.random_batch(
            batch_size, sequence_length, (fractions[1], 1))

        steps_per_epoch = int(epoch_scaling_factor*fractions[0]*
                              len(dataset.text8_data)/sequence_length/batch_size)
        validation_steps = int(steps_per_epoch*
                               validation_fraction/fractions[0])
        test_steps = int(steps_per_epoch*
                         test_fraction/fractions[0])

        train_kwargs = scope.setdefault('model_train_kwargs', {})
        train_kwargs['steps_per_epoch'] = steps_per_epoch
        train_kwargs['validation_steps'] = validation_steps
        scope['training_data_generator'] = scope['train_generator'] = train_data
        scope['generator_train_steps'] = steps_per_epoch
        scope['validation_data_generator'] = scope['validation_generator'] = val_data
        scope['generator_val_steps'] = validation_steps
        scope['test_data_generator'] = scope['test_generator'] = test_data
        scope['test_steps'] = test_steps
        scope['loss'] = 'sparse_categorical_crossentropy'
        scope['sequence_length'] = sequence_length
        scope['encoder'] = dataset.encode
        scope['decoder'] = dataset.decode

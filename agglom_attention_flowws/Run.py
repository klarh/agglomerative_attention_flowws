import json
import random
import re
import time

import flowws
from flowws import Argument as Arg
import gtar
import numpy as np
import keras
import keras.backend as K
import keras_tqdm
import keras_gtar

METRIC_MAP = {}

def metric(f):
    """Decorator for custom metrics in this module"""
    METRIC_MAP[f.__name__] = f
    return f

def intfloat(x):
    """int . float (for convenience in command-line specification)"""
    return int(float(x))

def maybe_setup_tensorflow():
    if keras.backend.backend() != 'tensorflow':
        return

    import tensorflow as tf

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    session = tf.Session(config=tf_config)
    K.set_session(session)

def maybe_set_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed + 1)
    if keras.backend.backend() == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(seed + 2)

@metric
def perplexity(y_true, y_pred):
    """
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))

@metric
def bpc(y_true, y_pred):
    """Bits per character metric, commonly used for compression-type tasks."""
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(cross_entropy)/np.log(2)

class TimingCallback(keras.callbacks.Callback):
    """Measure the time required for each epoch"""
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        logs['epoch_time'] = time.time() - self.starttime

class TimeLimitCallback(keras.callbacks.Callback):
    """End training after a certain amount of time has passed"""
    def __init__(self, limit):
        self.time_limit = self.parse_time(limit)

    def on_train_begin(self, *args, **kwargs):
        self.start_time = time.time()

    def on_epoch_end(self, *args, **kwargs):
        if time.time() - self.start_time > self.time_limit:
            self.model.stop_training = True

    @staticmethod
    def parse_time(limit):
        """Parse a human-readable time, like 8h3m, into seconds"""
        pattern = re.compile(r'(?P<number>\d+(\.\d+)?)(?P<unit>[smhd])(?P<rest>.*)')
        # convert units to seconds
        conversion = dict(s=1, m=60, h=60*60, d=60*60*24)

        remainder = limit
        result = 0

        while remainder:
            match = pattern.match(remainder)

            if match is None:
                raise ValueError('Can\'t parse time fragment "{}"'.format(remainder))

            unit = match.group('unit')
            number = float(match.group('number'))
            result += number*conversion[unit]

            remainder = match.group('rest')

        return int(result)

@flowws.add_stage_arguments
class Run(flowws.Stage):
    """Train and save a model"""

    ARGS = [
        Arg('metrics', '-m', [str], [],
            help='Metrics to compute and save'),
        Arg('epochs', '-e', intfloat, None,
            help='Number of epochs to train'),
        Arg('early_stopping', None, int, None,
            help='Patience for early stopping'),
        Arg('validation_split', None, float, .3,
            help='Fraction of training data to use as validation if no '
            'validation set is given'),
        Arg('optimizer', None, str, 'adadelta',
            help='Optimizer name to use'),
        Arg('optimizer_kwargs', None, [(str, eval)], [],
            help='Arguments for optimizer'),
        Arg('seed', '-s', int, None,
            help='Seed to use'),
        Arg('time_limit', '-t', str, None,
            help='Time limit to use (e.g. 8h)'),
    ]

    def run(self, scope, storage):
        maybe_setup_tensorflow()
        maybe_set_seed(self.arguments['seed'])

        model = scope['model']

        metrics = []

        for m in self.arguments['metrics']:
            metrics.append(METRIC_MAP.get(m, m))

        callbacks = [
            TimingCallback(),
            keras_tqdm.TQDMCallback(show_inner=False),
        ]

        if self.arguments.get('early_stopping', None):
            patience = self.arguments['early_stopping']
            callback = keras.callbacks.EarlyStopping(
                patience=patience, restore_best_weights=True, verbose=True)
            callbacks.append(callback)

        if self.arguments.get('time_limit', None):
            callback = TimeLimitCallback(self.arguments['time_limit'])
            callbacks.append(callback)

        optimizer_kwargs = dict(self.arguments.get('optimizer_kwargs', {}))
        optimizer_cls = getattr(keras.optimizers, self.arguments['optimizer'])
        optimizer = optimizer_cls(**optimizer_kwargs)

        model.compile(
            optimizer,
            loss=scope['loss'],
            metrics=metrics,
        )

        args = []
        kwargs = dict(scope.get('model_train_kwargs', {}))
        kwargs.update(dict(
            callbacks=callbacks,
            epochs=self.arguments['epochs'],
            verbose=False,
        ))

        use_fit_generator = False

        if 'training_data' in scope:
            args.extend(scope['training_data'])
            kwargs['batch_size'] = scope.get('batch_size', 32)
        elif 'training_data_generator' in scope:
            args.append(scope['training_data_generator'])
            use_fit_generator = True
        else:
            raise NotImplementedError()

        if 'validation_data' in scope:
            kwargs['validation_data'] = scope['validation_data']
        elif 'validation_data_generator' in scope:
            kwargs['validation_data'] = scope['validation_data_generator']
        elif self.arguments['validation_split'] > 0:
            kwargs['validation_split'] = self.arguments['validation_split']

        if use_fit_generator:
            history = model.fit_generator(*args, **kwargs)
        else:
            history = model.fit(*args, **kwargs)

        metadata = scope.get('metadata', {})

        with storage.open('dump.zip', 'wb', on_filesystem=True) as f:
            with keras_gtar.Trajectory(f.name, 'w') as traj:
                traj.save(model, str(history.epoch[-1]))

            with gtar.GTAR(f.name, 'a') as traj:
                for name, vals in history.history.items():
                    rec = gtar.Record(
                        '', name, '0', gtar.Behavior.Continuous,
                        gtar.Format.Float32, gtar.Resolution.Uniform)
                    traj.writeRecord(rec, vals)

                traj.writeStr('metadata.json', json.dumps(metadata))

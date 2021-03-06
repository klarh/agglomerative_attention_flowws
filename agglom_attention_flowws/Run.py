import contextlib
import json
import random
import os
import re
import time

import flowws
from flowws import Argument as Arg
import gtar
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
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

    tf.config.optimizer.set_jit(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

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
        self.triggered = False

    def on_epoch_end(self, *args, **kwargs):
        if time.time() - self.start_time > self.time_limit:
            self.model.stop_training = True
            self.triggered = True

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
        Arg('optimizer', None, str, 'Adadelta',
            help='Optimizer name to use'),
        Arg('optimizer_kwargs', None, [(str, eval)], [],
            help='Arguments for optimizer'),
        Arg('seed', '-s', int, None,
            help='Seed to use'),
        Arg('time_limit', '-t', str, None,
            help='Time limit to use (e.g. 8h)'),
        Arg('reduce_lr', None, int, None,
            help='Period over which to reduce learning rate'),
        Arg('reduce_lr_factor', None, float, .75,
            help='Factor by which to reduce learning rate'),
    ]

    def run(self, scope, storage):
        maybe_setup_tensorflow()
        maybe_set_seed(self.arguments.get('seed', None))

        model = self.get_model(scope, storage)

        initial_epoch = scope['last_epoch'] + 1

        if self.arguments['epochs'] <= initial_epoch:
            return

        metrics = []

        for m in self.arguments['metrics']:
            metrics.append(METRIC_MAP.get(m, m))

        callbacks = list(scope.setdefault('callbacks', []))
        callbacks.append(TimingCallback())

        early_stopping_callback = None
        if self.arguments.get('early_stopping', None):
            patience = self.arguments['early_stopping']
            early_stopping_callback = keras.callbacks.EarlyStopping(
                patience=patience, restore_best_weights=True, verbose=True)
            callbacks.append(early_stopping_callback)

        time_callback = None
        if self.arguments.get('time_limit', None):
            time_callback = TimeLimitCallback(self.arguments['time_limit'])
            callbacks.append(time_callback)

        if self.arguments.get('reduce_lr', None):
            reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
                factor=self.arguments['reduce_lr_factor'],
                patience=self.arguments['reduce_lr'])
            callbacks.append(reduce_lr_callback)

        optimizer_kwargs = dict(scope.get('optimizer_kwargs', {}))
        optimizer_kwargs.update(dict(self.arguments.get('optimizer_kwargs', {})))
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
            initial_epoch=initial_epoch,
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

        try:
            if use_fit_generator:
                history = model.fit_generator(*args, **kwargs)
            else:
                history = model.fit(*args, **kwargs)
        except KeyboardInterrupt:
            history = model.history

        test_evals = {}
        if 'test_data' in scope:
            values = model.evaluate(
                *scope['test_data'],
                verbose=False, batch_size=kwargs['batch_size'])
            test_evals.update(dict(zip(model.metrics_names, values)))
        elif 'test_data_generator' in scope:
            values = model.evaluate_generator(
                scope['test_data_generator'],
                verbose=False, steps=scope['test_steps'])
            test_evals.update(dict(zip(model.metrics_names, values)))

        metadata = scope.get('metadata', {})
        filename = scope.get('filename', 'dump.zip')
        final_epoch = history.epoch[-1]

        end_reason = None
        if final_epoch + 1 >= self.arguments['epochs']:
            end_reason = 'completed'
        elif time_callback is not None and time_callback.triggered:
            pass # we should resume training later
        elif early_stopping_callback is not None and early_stopping_callback.stopped_epoch:
            end_reason = 'early_stopping'

        with storage.open(filename, 'ab', on_filesystem=True) as f:
            gtar_mode = 'a' if os.stat(f.name).st_size > 0 else 'w'
            with keras_gtar.Trajectory(f.name, gtar_mode) as traj:
                traj.save(model, str(final_epoch))

            with gtar.GTAR(f.name, 'a') as traj:
                for name, vals in history.history.items():
                    rec = gtar.Record(
                        '', name, str(final_epoch), gtar.Behavior.Continuous,
                        gtar.Format.Float32, gtar.Resolution.Uniform)
                    traj.writeRecord(rec, vals)

                for name, vals in test_evals.items():
                    name = 'test_{}'.format(name)
                    rec = gtar.Record(
                        '', name, str(final_epoch), gtar.Behavior.Discrete,
                        gtar.Format.Float32, gtar.Resolution.Uniform)
                    traj.writeRecord(rec, vals)

                traj.writeStr('metadata.json', json.dumps(metadata))

                if end_reason is not None:
                    traj.writeStr('end_reason.txt', end_reason)

    def get_model(self, scope, storage):
        model = scope['model']

        filename = scope.get('filename', 'dump.zip')

        optimizer_kwargs = scope.setdefault('optimizer_kwargs', {})

        try:
            with contextlib.ExitStack() as stack:
                f = stack.enter_context(
                    storage.open(filename, 'rb', on_filesystem=True))

                with gtar.GTAR(f.name, 'r') as traj:
                    if traj.readStr('end_reason.txt'):
                        scope['last_epoch'] = self.arguments['epochs']
                        return model

                    for (_, lr_array) in traj.recordsNamed('lr'):
                        optimizer_kwargs['lr'] = lr_array[-1]

                traj = stack.enter_context(
                    keras_gtar.Trajectory(f.name))

                last_frame = int(traj.frames[-1])

                new_model = traj.load()
                # reuse the original model object to make sure the
                # architecture did not change
                model.set_weights(new_model.get_weights())

                scope['last_epoch'] = last_frame
        except FileNotFoundError:
            scope['last_epoch'] = -1

        return model

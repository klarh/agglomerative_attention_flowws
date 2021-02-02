import flowws
from flowws import Argument as Arg
import numpy as np
from tensorflow.keras import backend as K

METRIC_MAP = {}

def metric(f):
    """Decorator for custom metrics in this module"""
    METRIC_MAP[f.__name__] = f
    return f

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

@flowws.add_stage_arguments
class TextMetrics(flowws.Stage):
    ARGS = [
        Arg('metrics', '-m', [str], [],
            help='Metrics to compute and save'),
        ]

    def run(self, scope, storage):
        metrics = scope.setdefault('metrics', [])

        for m in self.arguments['metrics']:
            metrics.append(METRIC_MAP.get(m, m))

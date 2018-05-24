import os
import random
from collections import defaultdict
import pickle

class AveragedPerceptron(object):
    """
    A class for training a model for answering the questions.
    """

    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
        self.classes = set()
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def get_scores(self, features):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float)
        for feat, value in features.items():
            if isinstance(value, list):
                wordlist = value
                for i, w in enumerate(wordlist):
                    wf, wfvalue = w
                    if wf not in self.weights or wfvalue == 0:
                        continue
                    weights = self.weights[wf]
                    for label, weight in weights.items():
                        scores[label] += wfvalue * weight
            else:
                if feat not in self.weights or value == 0:
                    continue
                weights = self.weights[feat]
                for label, weight in weights.items():
                    scores[label] += value * weight
        return scores
    
    def get_scored_classes(self, features):
        scores = self.get_scores(features)
        return sorted(self.classes, key=lambda label: (scores[label], label), reverse=True)
    
    def predict(self, features):
        scores = self.get_scores(features)
        return max(self.classes, key=lambda label: (scores[label], label))

    def update(self, truth, guess, features):
        '''Update the feature weights.'''
        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            if isinstance(features[f], list):
                for w in features[f]:
                    wf = w[0]
                    weights = self.weights.setdefault(wf, {})
                    upd_feat(truth, wf, weights.get(truth, 0.0), 1.0)
                    upd_feat(guess, wf, weights.get(guess, 0.0), -1.0)
            else:
                weights = self.weights.setdefault(f, {})
                upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
                upd_feat(guess, f, weights.get(guess, 0.0), -1.0)
        return None

    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights
        return None

    def save(self, path):
        '''Save the pickled model weights.'''
        return pickle.dump((dict(self.weights), self.classes), open(path, 'wb'))

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = pickle.load(open(path))
        return None
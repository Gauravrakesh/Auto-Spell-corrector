from ngram import NGramBase
from config import *
import numpy as np
import pandas as pd
from collections import defaultdict
from math import log
from config import add_k
from config import stupid_backoff
from config import good_turing
from config import interpolation
from config import kneser_ney
from config import no_smoothing
from config import interpolation

class NoSmoothing(NGramBase):
    def __init__(self):
        super(NoSmoothing, self).__init__()
        self.update_config(no_smoothing)

    def get_probability(self, ngram, context):
        """Raw MLE probability estimation"""
        if self.context_counts[context] == 0:
            return 0.0
        return self.ngram_counts[ngram] / self.context_counts[context]

class AddK(NGramBase):
    def __init__(self):
        super(AddK, self).__init__()
        self.update_config(add_k)
        self.k = self.current_config['k']

    def get_probability(self, ngram, context):
        """Add-k smoothing"""
        numerator = self.ngram_counts[ngram] + self.k
        denominator = self.context_counts[context] + self.k * len(self.vocab)
        return numerator / denominator if denominator > 0 else 0.0

class StupidBackoff(NGramBase):
    def __init__(self):
        super(StupidBackoff, self).__init__()
        self.update_config(stupid_backoff)
        self.alpha = self.current_config['alpha']

    def get_probability(self, ngram, context):
        """Stupid backoff smoothing"""
        if self.context_counts[context] > 0:
            return self.ngram_counts[ngram] / self.context_counts[context]
        # Back off to (n-1)-gram
        return self.alpha * self.get_probability(ngram[1:], context[1:])

class GoodTuring(NGramBase):
    def __init__(self):
        super(GoodTuring, self).__init__()
        self.update_config(good_turing)
        self.count_of_counts = defaultdict(int)
        
    def fit(self, data):
        super().fit(data)
        # Calculate count-of-counts
        for count in self.ngram_counts.values():
            self.count_of_counts[count] += 1

    def get_probability(self, ngram, context):
        """Good-Turing smoothing"""
        count = self.ngram_counts[ngram]
        if count == 0:
            return self.count_of_counts[1] / (self.total_words * self.count_of_counts[0])
        
        # Compute smoothed count
        next_count = count + 1
        if self.count_of_counts[next_count] == 0:
            return count / self.total_words
            
        smoothed_count = (count + 1) * self.count_of_counts[next_count] / self.count_of_counts[count]
        return smoothed_count / self.total_words

class Interpolation(NGramBase):
    def __init__(self):
        super(Interpolation, self).__init__()
        self.update_config(interpolation)
        self.lambdas = self.current_config['lambdas']

    def get_probability(self, ngram, context):
        """Linear interpolation smoothing"""
        prob = 0.0
        for i in range(self.n):
            if i == 0:
                # Trigram probability
                if self.context_counts[context] > 0:
                    prob += self.lambdas[i] * (self.ngram_counts[ngram] / self.context_counts[context])
            elif i == 1:
                # Bigram probability
                bigram_context = context[1:]
                if self.context_counts[bigram_context] > 0:
                    prob += self.lambdas[i] * (self.ngram_counts[ngram[1:]] / self.context_counts[bigram_context])
            else:
                # Unigram probability
                prob += self.lambdas[i] * (self.ngram_counts[ngram[-1:]] / self.total_words)
        return prob

class KneserNey(NGramBase):
    def __init__(self):
        super(KneserNey, self).__init__()
        self.update_config(kneser_ney)
        self.discount = self.current_config['discount']
        self.continuation_counts = defaultdict(int)

    def fit(self, data):
        super().fit(data)
        # Calculate continuation counts
        for ngram in self.ngram_counts:
            self.continuation_counts[ngram[:-1]] += 1

    def get_probability(self, ngram, context):
        """Kneser-Ney smoothing"""
        if len(context) == 0:
            return self.continuation_counts[ngram] / sum(self.continuation_counts.values())
            
        count = self.ngram_counts[ngram]
        if count == 0:
            return self.get_probability(ngram[1:], context[1:])
            
        higher_order = max(0, count - self.discount) / self.context_counts[context]
        lower_order = (self.discount * self.continuation_counts[context] / self.context_counts[context]) * \
                     self.get_probability(ngram[1:], context[1:])
        
        return higher_order + lower_order

if __name__=="__main__":
    ns = NoSmoothing()
    ns.method_name()

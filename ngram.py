import numpy as np
import pandas as pd
from typing import List
import re
from collections import defaultdict
from math import log

class NGramBase:
    def __init__(self):
        """
        Initialize basic n-gram configuration.
        :param n: The order of the n-gram (e.g., 2 for bigram, 3 for trigram).
        :param lowercase: Whether to convert text to lowercase.
        :param remove_punctuation: Whether to remove punctuation from text.
        """
        self.current_config = {}
        self.n = 3  # Default to trigram
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.total_words = 0

    def method_name(self) -> str:
        return f"Method Name: {self.current_config['method_name']}"

    def fit(self, data: List[List[str]]) -> None:
        """
        Fit the n-gram model to the data.
        :param data: The input data. Each sentence is a list of tokens.
        """
        # Add start and end tokens and count n-grams
        for sentence in data:
            padded_sent = ['<s>'] * (self.n-1) + sentence + ['</s>']
            self.vocab.update(sentence)
            self.total_words += len(sentence)
            
            # Count n-grams and their contexts
            for i in range(len(padded_sent)-self.n+1):
                ngram = tuple(padded_sent[i:i+self.n])
                context = tuple(padded_sent[i:i+self.n-1])
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        :param text: The input text.
        :return: The list of tokens.
        """
        # Basic whitespace tokenization
        return text.strip().split()

    def prepare_data_for_fitting(self, data: List[str], use_fixed = False) -> List[List[str]]:
        """
        Prepare data for fitting.
        :param data: The input data.
        :return: The prepared data.
        """
        processed = []
        if not use_fixed:
            for text in data:
                processed.append(self.tokenize(self.preprocess(text)))
        else:
            for text in data:
                processed.append(self.fixed_tokenize(self.fixed_preprocess(text)))

        return processed

    def update_config(self, config) -> None:
        """
        Override the current configuration. You can use this method to update
        the config if required
        :param config: The new configuration.
        """
        self.current_config = config

    def preprocess(self, text: str) -> str:
        """
        Preprocess text before n-gram extraction.
        :param text: The input text.
        :return: The preprocessed text.
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_preprocess(self, text: str) -> str:
        """
        Removes punctuation and converts text to lowercase.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text by splitting at spaces.
        """
        return text.split()

    def perplexity(self, text: str) -> float:
        """
        Compute the perplexity of the model given the text.
        :param text: The input text.
        :return: The perplexity of the model.
        """
        tokens = self.tokenize(self.preprocess(text))
        padded_text = ['<s>'] * (self.n-1) + tokens + ['</s>']
        log_prob = 0
        count = 0
        
        for i in range(len(padded_text)-self.n+1):
            ngram = tuple(padded_text[i:i+self.n])
            context = tuple(padded_text[i:i+self.n-1])
            
            # Use simple maximum likelihood estimation
            if self.context_counts[context] > 0:
                prob = self.ngram_counts[ngram] / self.context_counts[context]
            else:
                prob = 1 / len(self.vocab)  # Simple smoothing
                
            log_prob += log(prob)
            count += 1
            
        return np.exp(-log_prob/count)

if __name__ == "__main__":
    tester_ngram = NGramBase()
    test_sentence = "This, is a ;test sentence."

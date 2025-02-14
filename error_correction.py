from smoothing_classes import *
from config import error_correction
import numpy as np
import pandas as pd
from typing import List, Set, Dict, Tuple
import re
from collections import defaultdict

class SpellingCorrector:
    def __init__(self):
        print("Initializing SpellingCorrector...")
        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']
        self.vocabulary = set()
        # Error model stores P(typo|correct_word)
        self.error_model = defaultdict(lambda: defaultdict(float))
        self.char_counts = defaultdict(int)  # For character frequency
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))

        # Initialize appropriate n-gram model for language modeling
        if self.internal_ngram_name == "NO_SMOOTH":
            self.internal_ngram = NoSmoothing()
        elif self.internal_ngram_name == "ADD_K":
            self.internal_ngram = AddK()
        elif self.internal_ngram_name == "STUPID_BACKOFF":
            self.internal_ngram = StupidBackoff()
        elif self.internal_ngram_name == "GOOD_TURING":
            self.internal_ngram = GoodTuring()
        elif self.internal_ngram_name == "INTERPOLATION":
            self.internal_ngram = Interpolation()
        elif self.internal_ngram_name == "KNESER_NEY":
            self.internal_ngram = KneserNey()

        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])

    def fit(self, data: List[str]) -> None:
        """
        Fit the spelling corrector model to the data.
        :param data: The input data.
        """
      
        # Process and fit language model
        processed_data = self.internal_ngram.prepare_data_for_fitting(data, use_fixed=True)
        self.internal_ngram.fit(processed_data)
        
        # Build vocabulary and character statistics
        for text in processed_data:
            self.vocabulary.update(text)
            for word in text:
                for char in word:
                    self.char_counts[char] += 1

        # Train error model using character frequencies and common error patterns
        self._train_error_model()

    def _train_error_model(self) -> None:
        """
        Train the error model using character frequencies and common error patterns.
        Estimates P(typo|correct_word) for various error types.
        """
        total_chars = sum(self.char_counts.values())
        char_probs = {c: count/total_chars for c, count in self.char_counts.items()}

        for word in self.vocabulary:
            # Generate potential errors
            edits = self._generate_edits(word)
            
            for error_type, candidates in edits.items():
                for candidate in candidates:
                    # Weight error probabilities based on error type and character frequencies
                    if error_type == 'deletion':
                        prob = 0.25 * char_probs.get(candidate[1], 0.01)
                    elif error_type == 'insertion':
                        prob = 0.25 * char_probs.get(candidate[1], 0.01)
                    elif error_type == 'substitution':
                        prob = 0.25 * char_probs.get(candidate[1], 0.01)
                    else:  # transposition
                        prob = 0.25  # Equal probability for all transpositions
                        
                    self.error_model[candidate[0]][word] += prob

            # Normalize probabilities
            total = sum(self.error_model[word].values())
            if total > 0:
                for w in self.error_model[word]:
                    self.error_model[word][w] /= total

    def _generate_edits(self, word: str) -> Dict[str, Set[Tuple[str, str]]]:
        """
        Generate all possible single-character edits for a word.
        Returns a dictionary of edit types to sets of (error, char) tuples.
        """
        edits = {
            'deletion': set(),
            'insertion': set(),
            'substitution': set(),
            'transposition': set()
        }
        
        for i in range(len(word) + 1):
            if i < len(word):
                # Deletions
                error = word[:i] + word[i+1:]
                edits['deletion'].add((error, word[i]))
                
                # Substitutions
                for c in self.alphabet:
                    if c != word[i]:
                        error = word[:i] + c + word[i+1:]
                        edits['substitution'].add((error, c))
                        
            # Insertions
            for c in self.alphabet:
                error = word[:i] + c + word[i:]
                edits['insertion'].add((error, c))
                
            # Transpositions
            if i < len(word) - 1:
                error = word[:i] + word[i+1] + word[i] + word[i+2:]
                edits['transposition'].add((error, word[i:i+2]))
                
        return edits

    def correct(self, text: List[str]) -> List[str]:
        """
        Correct misspellings using the noisy channel model.
        P(correction|error) ∝ P(error|correction) * P(correction)
        :param text: List of words potentially containing errors
        :return: List of corrected words
        """
        corrected_text = []
        n = len(text)
        
        for i in range(n):
            word = text[i].lower()
            
            # Skip if word is in vocabulary
            if word in self.vocabulary:
                corrected_text.append(word)
                continue
                
            # Generate candidates within edit distance 1
            candidates = set()
            for edit_type, edits in self._generate_edits(word).items():
                candidates.update(c[0] for c in edits)
            candidates = {w for w in candidates if w in self.vocabulary}
            
            if not candidates:
                corrected_text.append(word)
                continue
                
            # Find best candidate using noisy channel model
            best_prob = float('-inf')
            best_correction = word
            
            for candidate in candidates:
                # P(error|correction) from error model
                error_prob = self.error_model[word][candidate]
                
                # P(correction) from language model using context
                context = text[max(0, i-2):i]
                lm_prob = self.internal_ngram.score([context + [candidate]])
                
                # Combine probabilities in log space
                total_prob = np.log(error_prob + 1e-10) + lm_prob
                
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_correction = candidate
                    
            corrected_text.append(best_correction)
            
        assert len(corrected_text) == len(text), "Output length must match input length"
        return corrected_text

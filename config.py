# N-gram configuration
ngram_config = {
    "n": 3,  # Use trigrams for better context
}

# Smoothing configurations
no_smoothing = {
    "method_name": "NO_SMOOTH"
    # Uses default n=3 from ngram_config
}

add_k = {
    "method_name": "ADD_K", 
    "k": 0.1,  
    "n" : 3, 
}

stupid_backoff = {
    "method_name": "STUPID_BACKOFF",
    "alpha": 0.4,  # Standard value from literature
     "n" : 3,
}

good_turing = {
    "method_name": "GOOD_TURING",
    "n" : 3,
    # Uses default n=3 from ngram_config
}

interpolation = {
    "method_name": "INTERPOLATION",
    "lambdas": [0.1, 0.3, 0.6],  # Weights for trigram, bigram, unigram
    "n" : 3,
    # Uses default n=3 from ngram_config
}

kneser_ney = {
    "method_name": "KNESER_NEY",
    "discount": 0.75,  # Discount parameter for higher order n-grams
    "n" : 3,
    # Uses default n=3 from ngram_config
}

# Error correction configuration
error_correction = {
    # Internal n-gram model configuration
    "internal_ngram_best_config": {
        "method_name": "KNESER_NEY",
        "discount": 0.75,
         "n" : 3,
    },
    
    # Error model parameters
    "error_weights": {
        "deletion": 0.25,
        "insertion": 0.25,
        "substitution": 0.25,
        "transposition": 0.25
    },
    
    # Error correction parameters
    "min_probability": 1e-10,
    "max_edit_distance": 1,
    "max_candidates": 5,
    "max_iterations": 10,
    "context_window": 2,
    "char_smoothing": 0.01,
    
    # Processing flags
    "use_fixed_preprocess": True,
    "use_fixed_tokenize": True,
    "use_fixed_ngram": True,
    "use_fixed_error_correction": True,
    "use_fixed_error_weights": True,
    "use_fixed_min_probability": True
}

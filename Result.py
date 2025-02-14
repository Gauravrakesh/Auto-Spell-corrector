from error_correction import SpellingCorrector
import re
def read_training_file("./data/train1.txt"):
    """Read training data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        training_data = f.readlines()
    
    # Preprocess training sentences
    training_data = [sentence.strip().lower().split() for sentence in training_data]
    return training_data

def read_test_file("./data/misspelling_public.txt"):
    """Read test data in format: <CORRECT TEXT> && <INCORRECT TEXT>"""
    correct_sentences = []
    incorrect_sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split on && and strip whitespace
            correct, incorrect = line.split('&&')
            correct = correct.strip().strip('<>').strip()
            incorrect = incorrect.strip().strip('<>').strip()
            
            # Convert to lowercase and split into words
            correct_sentences.append(correct.lower().split())
            incorrect_sentences.append(incorrect.lower().split())
    
    return correct_sentences, incorrect_sentences


def main():
    # Initialize spelling corrector
    corrector = SpellingCorrector()
    
    # Read and fit training data
    print("Reading training data...")
    training_data = read_training_file('train1.txt')
    
    print("Fitting the model...")
    corrector.fit(training_data)
    
    # Read test data
    print("Reading test data...")
    correct_sentences, incorrect_sentences = read_test_file('./data/misspelling_public.txt')
    
    # Correct each test sentence
    print("Correcting test sentences...")
    corrected_sentences = []
    for incorrect_sent in incorrect_sentences:
        corrected = corrector.correct(incorrect_sent)
        corrected_sentences.append(corrected)
    
    # Calculate and display accuracy
    total_words = 0
    correct_words = 0
    total_sentences = len(correct_sentences)
    perfect_sentences = 0
    
    print("\nResults:")
    print("-" * 50)
    
    for i, (orig, corr, true) in enumerate(zip(incorrect_sentences, 
                                              corrected_sentences, 
                                              correct_sentences)):
        # Print each sentence and its correction
        print(f"\nExample {i+1}:")
        print(f"Original:  {' '.join(orig)}")
        print(f"Corrected: {' '.join(corr)}")
        print(f"True:      {' '.join(true)}")
        
        # Calculate accuracy
        for o, c, t in zip(orig, corr, true):
            if o != t:  # only count words that needed correction
                if c == t:
                    correct_words += 1
                total_words += 1
        
        if ' '.join(corr) == ' '.join(true):
            perfect_sentences += 1
            

     # Print accuracy metrics
    print("\nAccuracy Metrics:")
    print("-" * 50)
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    sentence_accuracy = perfect_sentences / total_sentences
    
    print(f"Word-level accuracy: {word_accuracy:.2%}")
    print(f"Sentence-level accuracy: {sentence_accuracy:.2%}")
    print(f"Words corrected: {correct_words}/{total_words}")
    print(f"Perfect sentences: {perfect_sentences}/{total_sentences}")

if __name__ == "__main__":
    main()
        
            
        
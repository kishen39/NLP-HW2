import numpy as np
from collections import defaultdict

class BigramLanguageModel:
    def __init__(self):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        
    def train(self, corpus):
        """Train the model on the given corpus"""
        # Process each sentence in the corpus
        for sentence in corpus:
            tokens = sentence.split()
            
            # Update unigram counts and vocabulary
            for token in tokens:
                self.unigram_counts[token] += 1
                self.vocab.add(token)
            
            # Update bigram counts
            for i in range(len(tokens) - 1):
                current_word = tokens[i]
                next_word = tokens[i + 1]
                self.bigram_counts[current_word][next_word] += 1
    
    def calculate_bigram_probability(self, word1, word2):
        """Calculate MLE probability for bigram (word1, word2)"""
        if self.unigram_counts[word1] == 0:
            return 0.0  # If word1 never appeared, probability is 0
        
        return self.bigram_counts[word1][word2] / self.unigram_counts[word1]
    
    def sentence_probability(self, sentence):
        """Calculate probability of a given sentence"""
        tokens = sentence.split()
        
        # Handle edge cases
        if len(tokens) < 2:
            return 0.0
        
        probability = 1.0
        
        # Calculate probability for each bigram in the sentence
        for i in range(len(tokens) - 1):
            current_word = tokens[i]
            next_word = tokens[i + 1]
            
            bigram_prob = self.calculate_bigram_probability(current_word, next_word)
            probability *= bigram_prob
            
            # Debug print for each bigram
            print(f"P({next_word}|{current_word}) = {bigram_prob:.4f}")
        
        return probability
    
    def print_counts(self):
        """Print unigram and bigram counts"""
        print("=== UNIGRAM COUNTS ===")
        for word, count in sorted(self.unigram_counts.items()):
            print(f"{word}: {count}")
        
        print("\n=== BIGRAM COUNTS ===")
        for word1 in sorted(self.bigram_counts.keys()):
            for word2 in sorted(self.bigram_counts[word1].keys()):
                count = self.bigram_counts[word1][word2]
                print(f"({word1}, {word2}): {count}")
    
    def print_probabilities(self):
        """Print all bigram probabilities"""
        print("\n=== BIGRAM PROBABILITIES ===")
        for word1 in sorted(self.bigram_counts.keys()):
            total = self.unigram_counts[word1]
            for word2 in sorted(self.bigram_counts[word1].keys()):
                count = self.bigram_counts[word1][word2]
                prob = count / total
                print(f"P({word2}|{word1}) = {count}/{total} = {prob:.4f}")

def main():
    # Training corpus
    corpus = [
        "<s> I love NLP </s>",
        "<s> I love deep learning </s>", 
        "<s> deep learning is fun </s>"
    ]
    
    # Initialize and train the model
    model = BigramLanguageModel()
    model.train(corpus)
    
    # Print counts and probabilities
    model.print_counts()
    model.print_probabilities()
    
    # Test sentences
    test_sentences = [
        "<s> I love NLP </s>",
        "<s> I love deep learning </s>"
    ]
    
    print("\n=== SENTENCE PROBABILITIES ===")
    probabilities = []
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n--- Sentence {i}: '{sentence}' ---")
        prob = model.sentence_probability(sentence)
        probabilities.append(prob)
        print(f"Overall sentence probability: {prob:.6f}")
    
    # Determine which sentence the model prefers
    print("\n=== MODEL PREFERENCE ===")
    if probabilities[0] > probabilities[1]:
        print(f"The model prefers: '{test_sentences[0]}'")
        print(f"Reason: Higher probability ({probabilities[0]:.6f} > {probabilities[1]:.6f})")
    elif probabilities[1] > probabilities[0]:
        print(f"The model prefers: '{test_sentences[1]}'")
        print(f"Reason: Higher probability ({probabilities[1]:.6f} > {probabilities[0]:.6f})")
    else:
        print("Both sentences have equal probability")

if __name__ == "__main__":
    main()
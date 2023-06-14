"""
Simple demo of a text generator using a Markov model with words as tokens.
GitHub: https://github.com/Mattis-Schulte/markov-chain-text-generation
"""

import numpy as np
import pickle
import re
from collections import defaultdict, Counter, deque
from time import time


class MarkovChainGenerator:
    def __init__(self, n=2):
        self.n = n
        self.probabilities = None

    def calculate_markov_chain(self, data):
        # Tokenize the input data
        tokens = self.tokenize(data)

        # Create a defaultdict of Counters to store the transition counts
        transitions = defaultdict(Counter)
        
        # Maintain a deque of previous tokens with a maximum length of n
        prev_tokens = deque(maxlen=self.n)
        
        # Iterate over the tokens and update the transition counts
        for token in tokens:
            if len(prev_tokens) == self.n:
                ngram = ' '.join(prev_tokens)
                transitions[ngram][token] += 1
            prev_tokens.append(token)

        # Calculate the probabilities based on the transition counts
        self.probabilities = {}
        for ngram, counter in transitions.items():
            total_count = sum(counter.values())
            self.probabilities[ngram] = {next_token: count / total_count for next_token, count in counter.items()}

    def tokenize(self, data):
        # Tokenize the data into words and remove punctuation
        data = re.sub(r'[^\w\s]', '', data)
        data = re.sub(r'\s+', ' ', data.strip())

        return data.split()

    def train_from_file(self, input_filename, output_filename):
        # Read the data from the input file
        with open(input_filename, 'r', encoding='utf8') as file:
            data = file.read()

        # Preprocessing: convert to lowercase
        data = data.lower()

        # Train the Markov chain model
        print(f'Training from {input_filename}...')
        start_time = time()
        self.calculate_markov_chain(data)
        print(f'Training completed in {time() - start_time:.2f} seconds.')

        # Write the probabilities to an output file
        print(f'Writing output to {output_filename}...')
        start_time = time()
        with open(output_filename, 'wb') as file:
            pickle.dump(self.probabilities, file)
        print(f'Output written to {output_filename} in {time() - start_time:.2f} seconds.')

    def read_probabilities(self, filename):
        # Read the probabilities from a file
        print(f'Reading probabilities from {filename}...')
        start_time = time()
        with open(filename, 'rb') as file:
            self.probabilities = pickle.load(file)
        print(f'Probabilities read from {filename} in {time() - start_time:.2f} seconds.')

    def generate_text(self, length):
        # Generate text using the trained Markov chain model
        print('Starting text generation...')
        if self.probabilities is None:
            raise ValueError('Markov chain is not trained.')

        # Choose a random starting ngram
        current_ngram = np.random.choice(list(self.probabilities.keys())).split()
        generated_text = current_ngram[:]
        print(' '.join(generated_text), end=' ', flush=True)

        # Generate text until the desired length is reached
        while len(generated_text) < length:
            if ' '.join(current_ngram) not in self.probabilities:
                break

            # Choose the next token based on the probabilities
            next_token = np.random.choice(list(self.probabilities[' '.join(current_ngram)].keys()), p=list(self.probabilities[' '.join(current_ngram)].values()))
            # Append the next token to the generated text
            generated_text.append(next_token)
            
            # Update the current ngram for the next iteration
            current_ngram = generated_text[-self.n:]

            # Print the last generated token
            print(generated_text[-1], end=' ', flush=True)

        return ' '.join(generated_text)


if __name__ == '__main__':
    generator = MarkovChainGenerator(n=3)
    generator.train_from_file('training_data_input.txt', 'training_data_output.pkl')
    # generator.read_probabilities('training_data_output.pkl')

    while True:
        try:
            t_num = input('\nNumber of tokens to generate: ')
            generator.generate_text(int(t_num))
        except ValueError:
            print('Invalid input. Please enter a valid integer.')
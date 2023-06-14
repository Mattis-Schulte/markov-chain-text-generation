"""
Simple demo of a text generator using a Markov model with single characters as tokens.
GitHub: https://github.com/Mattis-Schulte/markov-chain-text-generation
"""

import numpy as np
import pickle
from collections import defaultdict, Counter, deque
from time import time


class MarkovChainGenerator:
    def __init__(self, n=5):
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
                ngram = ''.join(prev_tokens)
                transitions[ngram][token] += 1
            prev_tokens.append(token)

        # Calculate the probabilities based on the transition counts
        self.probabilities = {}
        for ngram, counter in transitions.items():
            total_count = sum(counter.values())
            self.probabilities[ngram] = {next_token: count / total_count for next_token, count in counter.items()}

    def tokenize(self, data):
        # Tokenize the data into characters
        return iter(data)

    def train_from_file(self, input_filename, output_filename):
        # Read the data from the input file
        with open(input_filename, 'r', encoding='utf8') as file:
            data = file.read()

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
        current_ngram = ''.join(np.random.choice(list(self.probabilities.keys())))
        generated_text = current_ngram
        print(generated_text, end='', flush=True)

        # Generate text until the desired length is reached
        while len(generated_text) < length:
            if current_ngram not in self.probabilities:
                break

            # Choose the next token based on the probabilities
            next_token = np.random.choice(
                list(self.probabilities[current_ngram].keys()),
                p=list(self.probabilities[current_ngram].values())
            )
            # Append the next token to the generated text
            generated_text += next_token

            # Update the current ngram for the next iteration
            current_ngram = generated_text[-self.n:]

            # Print the last generated token
            print(generated_text[-1], end='', flush=True)

        return generated_text


if __name__ == '__main__':
    generator = MarkovChainGenerator(n=8)
    generator.train_from_file('tiny_shakespeare.txt', 'tiny_shakespeare.pkl')
    # generator.read_probabilities('tiny_shakespeare.pkl')

    while True:
        try:
            t_num = input('\nNumber of tokens to generate: ')
            generator.generate_text(int(t_num))
        except ValueError:
            print('Invalid input. Please enter a valid integer.')
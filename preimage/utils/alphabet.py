__author__ = "amelie"

import warnings
from itertools import product

import numpy as np

from preimage.exceptions.n_gram import InvalidNGramLengthError


class Alphabet:
    latin = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
             "v", "w", "x", "y", "z"]
    dna = ["A", "C", "G", "T"]


def get_n_gram_to_index(alphabet, n):
    n_grams = get_n_grams(alphabet, n)
    indexes = np.arange(len(n_grams))
    n_gram_to_index = dict(zip(n_grams, indexes))
    return n_gram_to_index


def get_index_to_n_gram(alphabet, n):
    n_grams = get_n_grams(alphabet, n)
    indexes = np.arange(len(n_grams))
    index_to_n_gram = dict(zip(indexes, n_grams))
    return index_to_n_gram


def get_n_grams(alphabet, n):
    n = int(n)
    if n <= 0:
        raise InvalidNGramLengthError(n)
    n_grams = ["".join(n_gram) for n_gram in product(alphabet, repeat=n)]
    return n_grams


def transform_strings_to_integer_lists(Y, alphabet):
    letter_to_int = get_n_gram_to_index(alphabet, 1)
    n_examples = np.array(Y).shape[0]
    max_length = np.max([len(y) for y in Y])
    Y_int = np.zeros((n_examples, max_length), dtype=np.int8) - 1
    for y_index, y in enumerate(Y):
        for letter_index, letter in enumerate(y):
            Y_int[y_index, letter_index] = letter_to_int[letter]
    return Y_int


def unique_dna_n_gram(n):
    """Generate a unique set of DNA n-grams that exclude reverse compliments.

    Parameters
    ----------
    n : int
        n-gram length

    Returns
    -------
    ngrams : list[str]
        Uniqe set of DNA n-grams
    """
    ngrams = set()
    for gram in get_n_grams(Alphabet.dna, n):
        # Above function generates all n-grams, check and make sure the reverse compliment is not already in the set
        if reverse_compliment(gram) not in ngrams:
            ngrams.add(gram)

    ngrams = list(ngrams)
    return ngrams


def transform_dna_to_pentamer_integer_lists(Y, alphabet):
    """Convert a DNA sequence into a list of ints, using a sliding window of 5. If a pentamer is not in the alphabet, then its reverse compliment must be in the alphabet.

    Parameters
    ----------
    Y : array-like
        The DNA sequences to convert.
    alphabet : array, shape = [512, ]
        Corresponding to the 512 DNA pentamers with unique DNA shape properties.

    Returns
    -------
    Y_int : array, shape = [n_Y, longest_Y - 4]
        The integer values of pentamers in each sequence. There are 4 fewer columns than the longest sequence because a sliding window is used.
    """
    window_size = 5
    pentamer_to_int = get_n_gram_to_index(alphabet, 1)
    n_pentamers = len(pentamer_to_int.keys())
    n_examples = np.array(Y).shape[0]
    max_length = np.max([len(y) for y in Y]) - window_size + 1

    # Initialize the array with values of -1, which correspond to no pentamer, i.e. the end of the sequence
    Y_int = np.full((n_examples, max_length), -1, dtype=np.int16)
    for y_index, y in enumerate(Y):
        # Indexing for the beginning of each pentamer
        for letter_index in range(len(y) - window_size + 1):
            pentamer = y[letter_index:letter_index+window_size]
            if pentamer in pentamer_to_int.keys():
                Y_int[y_index, letter_index] = pentamer_to_int[pentamer]
            # If reverse complimentation is necessary, add the number of pentamers in the alphabet to the int value
            # of the reverse compliment. This will help with handling edge cases in computing the kernel.
            else:
                pentamer = reverse_compliment(pentamer)
                Y_int[y_index, letter_index] = pentamer_to_int[pentamer] + n_pentamers

    return Y_int


def reverse_compliment(sequence):
    compliment_dict = {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A"
    }
    new_sequence = ""
    for base in sequence[::-1]:
        if base in compliment_dict.keys():
            new_sequence += compliment_dict[base]
        else:
            warnings.warn(f"Didn't recognize nucleotide {base}, using an N to reverse compliment.")
            new_sequence += "N"

    return new_sequence

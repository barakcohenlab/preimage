__author__ = 'amelie'

import numpy

from preimage.inference.bound_calculator import OCRMinBoundCalculator, MaxBoundCalculator, PeptideMinBoundCalculator
from preimage.inference.node_creator import NodeCreator
from preimage.utils.position import compute_position_weights
from preimage.utils.alphabet import get_n_gram_to_index, get_n_grams


def get_n_gram_node_creator(n, graph, graph_weights, y_length, n_gram_to_index, n_grams):
    max_bound = MaxBoundCalculator(n, graph, graph_weights.reshape(1, -1), n_gram_to_index)
    min_bound = OCRMinBoundCalculator(n, numpy.ones(y_length), n_grams)
    node_creator = NodeCreator(min_bound, max_bound, n_grams)
    return node_creator


def get_gs_node_creator(n, graph, graph_weights, y_length, n_gram_to_index, n_grams, sigma_position):
    position_weights = compute_position_weights(0, y_length, sigma_position)
    max_bound = MaxBoundCalculator(n, graph, graph_weights, n_gram_to_index)
    min_bound = OCRMinBoundCalculator(n, position_weights, n_grams)
    node_creator = NodeCreator(min_bound, max_bound, n_grams)
    return node_creator


def get_gs_similarity_node_creator(alphabet, n, graph, graph_weights, y_length, gs_kernel):
    n_gram_to_index = get_n_gram_to_index(alphabet, n)
    letter_to_index = get_n_gram_to_index(alphabet, 1)
    n_grams = get_n_grams(alphabet, n)
    min_bound = PeptideMinBoundCalculator(n, len(alphabet), n_grams, letter_to_index, y_length, gs_kernel)
    max_bound = MaxBoundCalculator(n, graph, graph_weights, n_gram_to_index)
    node_creator = NodeCreator(min_bound, max_bound, n_grams)
    return node_creator
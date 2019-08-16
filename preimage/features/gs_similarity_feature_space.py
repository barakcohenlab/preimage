__author__ = 'amelie, rfriedman22'

import numpy as np
from preimage.features.gs_similarity_weights import compute_gs_similarity_weights, compute_ngram_gs_similarity_weights
from preimage.utils.alphabet import get_n_grams


# Shouldn't label this as "feature-space" since we don't use a sparse matrix representation here.
class GenericStringSimilarityFeatureSpace:
    """Output space for the Generic String kernel with position and n-gram similarity.

    Doesn't use a sparse matrix representation because it takes in account the similarity between the n-grams.
    This is used to compute the weights of the graph during the inference phase.

    Attributes
    ----------
    n_max : int
        N-gram length.
    n_min : int
        Minimum sub n-gram length.
    is_normalized : bool
        True if the feature space should be normalized, False otherwise.
    max_train_length : int
        Length of the longest string in the training dataset.
    gs_kernel : GenericStringKernel
        Generic string kernel.
    """

    def __init__(self, alphabet, n_max, Y, is_normalized, gs_kernel, n_min=1):
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.is_normalized = is_normalized
        self.gs_kernel = gs_kernel
        # Use the kernel to transform input sequences, because the transformation depends on the original property
        # similarity matrix.
        self._Y_int = self.gs_kernel.transform(Y)
        self._y_lengths = np.array([len(y) - self.n_min + 1 for y in Y])
        self.max_train_length = np.max(self._y_lengths)
        self._n_grams_int = self.gs_kernel.transform(get_n_grams(alphabet, self.n_max))
        self._n_gram_similarity_matrix = gs_kernel.get_alphabet_similarity_matrix()
        if is_normalized:
            self._normalization = np.sqrt(gs_kernel.element_wise_kernel(Y))

    def compute_weights(self, y_weights, y_length):
        """Compute the inference graph weights

        Parameters
        ----------
        y_weights :  array, [n_samples]
            Weight of each training example.
        y_length : int
            Length of the string to predict.

        Returns
        -------
        gs_weights : [len(alphabet)**n, y_n_gram_count * len(alphabet)**n]
            Weight of each n-gram at each position, where y_n_gram_count is the number of n-gram in y_length.
        """
        normalized_weights = np.copy(y_weights)
        y_length = y_length - self.n_max + 1
        max_length = max(y_length, self.max_train_length)
        if self.is_normalized:
            normalized_weights *= 1. / self._normalization
        n_partitions = y_length
        position_matrix = self.gs_kernel.get_position_matrix(max_length)
        # Determining which C function to call depends on the dtype of _n_grams_int, which is determined by which
        # type of GS kernel is used.
        if self.gs_kernel.is_amino_acid():
            gs_weights = compute_gs_similarity_weights(n_partitions, self._n_grams_int, self._Y_int, normalized_weights,
                                                       self._y_lengths, position_matrix, self._n_gram_similarity_matrix,
                                                       self.n_min)
        elif self.gs_kernel.is_dna_kmer():
            gs_weights = compute_ngram_gs_similarity_weights(n_partitions, self._n_grams_int, self._Y_int,
                                                             normalized_weights, self._y_lengths, position_matrix,
                                                             self._n_gram_similarity_matrix)
        else:
            raise NotImplementedError("GS similarity weights not implemented for this kernel.")

        return np.array(gs_weights)
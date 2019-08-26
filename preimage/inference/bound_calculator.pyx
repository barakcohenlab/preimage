from libc.math cimport sqrt

cimport numpy as np
import numpy as np

from node cimport MaxNode


cdef class BoundCalculator:
    cdef Bound compute_bound(self, str  y, FLOAT64_t parent_real_value, int final_length):
         raise NotImplementedError('Subclasses should implement this method')

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        raise NotImplementedError('Subclasses should implement this method')

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        raise NotImplementedError('Subclasses should implement this method')

    # For unit tests only
    def compute_bound_python(self, str  y, FLOAT64_t parent_real_value, int final_length):
        return self.compute_bound(y, parent_real_value, final_length)

    # For unit tests only
    def get_start_node_real_values_python(self, int final_length):
        return np.array(self.get_start_node_real_values(final_length))

    # For unit tests only
    def get_start_node_bounds_python(self, int final_length):
        return np.array(self.get_start_node_bounds(final_length))


cdef class MaxBoundCalculator(BoundCalculator):
    def __init__(self, n_max, graph, graph_weights, n_gram_to_index):
        self.n_max = n_max
        self.graph = graph
        self.graph_weights = graph_weights
        self.n_gram_to_index = n_gram_to_index

    cdef Bound compute_bound(self, str  y, FLOAT64_t parent_real_value, int final_length):
        cdef int max_partition_index = final_length - self.n_max
        cdef int partition_index = max_partition_index - (len(y) - self.n_max)
        cdef int graph_weight_partition_index = min(self.graph_weights.shape[0]-1, partition_index)
        cdef int n_gram_index = self.n_gram_to_index[y[0:self.n_max]]
        cdef Bound max_bound
        max_bound.real_value = self.graph_weights[graph_weight_partition_index, n_gram_index] + parent_real_value
        max_bound.bound_value = self.graph[partition_index, n_gram_index] + parent_real_value
        return max_bound

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        cdef int max_partition_index, graph_weight_partition_index
        max_partition_index = final_length - self.n_max
        graph_weight_partition_index = min(self.graph_weights.shape[0] - 1, max_partition_index)
        return self.graph_weights[graph_weight_partition_index]

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        return self.graph[final_length - self.n_max]


cdef class OCRMinBoundCalculator(BoundCalculator):
    def __init__(self, n, position_weights, n_grams):
        self.n = n
        self.position_weights = position_weights
        self.n_grams = n_grams

    # In our experiments, we consider that YY' is zero since |A| > |y|
    cdef Bound compute_bound(self, str y, FLOAT64_t parent_real_value, int final_length):
        cdef FLOAT64_t gs_similarity = parent_real_value + self.gs_similarity_new_n_gram(y)
        cdef FLOAT64_t y_y_similarity = final_length - len(y)
        cdef Bound bound
        bound.bound_value = y_y_similarity + gs_similarity
        bound.real_value = gs_similarity
        return bound

    # Similarity only for the n-gram comparison (not blended)
    cdef FLOAT64_t gs_similarity_new_n_gram(self, str y):
        cdef int i
        cdef FLOAT64_t similarity = 0.
        for i in range(1, len(y) - self.n + 1):
            if y[0:self.n] == y[i:i + self.n]:
                similarity += self.position_weights[i]
        return 1 + 2 * similarity

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        return np.ones(len(self.n_grams))

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        return np.ones(len(self.n_grams)) * (final_length - self.n + 1)


cdef class MinBoundCalculator(BoundCalculator):
    def __init__(self, alphabet_length, n_grams, final_length, gs_kernel):
        self.alphabet_length = alphabet_length
        self.similarity_matrix = gs_kernel.get_alphabet_similarity_matrix()
        self.position_matrix = gs_kernel.get_position_matrix(final_length)
        self.start_node_real_values = gs_kernel.element_wise_kernel(n_grams)
        self.y_y_bounds = self.precompute_y_y_bound_for_each_length(final_length)
        self.start_node_bound_values = self.precompute_start_node_bounds(final_length, n_grams)

    cdef FLOAT64_t[::1] precompute_y_y_bound_for_each_length(self, int max_length):
        raise NotImplementedError("Subclass should implement this method.")

    cdef FLOAT64_t[::1] precompute_start_node_bounds(self, int final_length, list n_grams):
        raise NotImplementedError("Subclass should implement this method.")

    cdef FLOAT64_t compute_y_y_prime_bound(self, str y, int y_start_index):
        raise NotImplementedError("Subclass should implement this method.")

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        return self.start_node_real_values

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        return self.start_node_bound_values


cdef class PeptideMinBoundCalculator(MinBoundCalculator):
    def __init__(self, n_max, alphabet_length, n_grams, letter_to_index, final_length, gs_kernel, n_min=1):
        self.n_max = n_max
        self.n_min = n_min
        self.letter_to_index = letter_to_index
        super(PeptideMinBoundCalculator, self).__init__(alphabet_length, n_grams, final_length, gs_kernel)

    cdef FLOAT64_t[::1] precompute_y_y_bound_for_each_length(self, int max_length):
        cdef FLOAT64_t[::1] y_y_bounds = np.zeros(max_length, dtype=np.float64)
        cdef FLOAT64_t min_similarity = np.min(self.similarity_matrix)
        cdef FLOAT64_t length_bound, current_similarity
        cdef int length, j, k, l
        # For all possible prefix lengths (d in equation 16)
        for length in range(1, max_length):
            length_bound = 0
            # Position 1
            for j in range(length):
                # Position 2
                for k in range(j + 1, length):
                    current_similarity = 1.
                    # All n-gram lengths
                    for l in range(self.n_min - 1, self.n_max):
                        current_similarity *= min_similarity
                        length_bound += self.position_matrix[j, k] * current_similarity
            # self.n_max * length cases where j == k so S(.) = 1.
            y_y_bounds[length] =  self.n_max * length + 2 * length_bound
        return y_y_bounds

    cdef FLOAT64_t[::1] precompute_start_node_bounds(self, int final_length, list n_grams):
        cdef FLOAT64_t[::1] bounds = np.empty(len(n_grams), dtype=np.float64)
        cdef FLOAT64_t y_y_prime_bound
        cdef str n_gram
        cdef int i
        cdef int n_gram_start_index = final_length - self.n_max
        for i, n_gram in enumerate(n_grams):
            y_y_prime_bound = self.compute_y_y_prime_bound(n_gram, n_gram_start_index)
            bounds[i] = self.start_node_real_values[i] + self.y_y_bounds[n_gram_start_index] + 2 * y_y_prime_bound
        return bounds

    cdef Bound compute_bound(self, str y, FLOAT64_t parent_real_value, int final_length):
        cdef FLOAT64_t gs_similarity = parent_real_value + self.gs_similarity_new_n_gram(y)
        cdef FLOAT64_t y_y_similarity = self.y_y_bounds[final_length - len(y) - self.n_min + 1]
        cdef FLOAT64_t y_y_prime_similarity = self.compute_y_y_prime_bound(y, final_length - len(y) - self.n_min + 1)
        cdef Bound bound
        bound.bound_value = gs_similarity + y_y_similarity + 2 * y_y_prime_similarity
        bound.real_value = gs_similarity
        return bound

    cdef FLOAT64_t gs_similarity_new_n_gram(self, str y):
        cdef int i, l, max_length, index_one, index_two
        cdef FLOAT64_t current_similarity, n_gram_similarity
        cdef FLOAT64_t similarity = 0.
        for i in range(1, len(y) - self.n_min + 1):
            max_length = min(self.n_max, len(y) - i)
            current_similarity = 1.
            n_gram_similarity = 0.
            for l in range(max_length):
                index_one = self.letter_to_index[y[l]]
                index_two = self.letter_to_index[y[i + l]]
                current_similarity *= self.similarity_matrix[index_one, index_two]
                n_gram_similarity += current_similarity
            similarity += self.position_matrix[0, i] * n_gram_similarity
        return self.n_max + 2 * similarity

    cdef FLOAT64_t compute_y_y_prime_bound(self, str y, int y_start_index):
        # Equation 17 from the ICML paper
        cdef np.ndarray[FLOAT64_t, ndim=2] similarity_matrix = np.asarray(self.similarity_matrix)
        cdef int i, n_gram_length
        cdef FLOAT64_t y_y_prime_bound = 0
        # l in equation 17
        for n_gram_length in range(self.n_min, self.n_max + 1):
            # i in equation 17
            for i in range(y_start_index):
                y_y_prime_bound += self.compute_n_gram_y_y_prime_bound(n_gram_length, i, y, y_start_index,
                                                                       similarity_matrix)
        return y_y_prime_bound

    cdef FLOAT64_t compute_n_gram_y_y_prime_bound(self, int n_gram_length, int n_gram_index, str y, int y_start_index,
                                                  np.ndarray[FLOAT64_t, ndim=2] similarity_matrix):
        cdef int i, l, letter_index
        cdef str letter
        cdef np.ndarray[FLOAT64_t, ndim=1] n_gram_scores, final_scores
        final_scores = np.zeros(self.alphabet_length ** n_gram_length)
        # j in equation 17
        for i in range(len(y) - n_gram_length + 1):
            n_gram_scores = np.ones(self.alphabet_length ** n_gram_length)
            for l in range(n_gram_length):
                letter_index = self.letter_to_index[y[i + l]]
                n_gram_scores *= self.transform_letter_scores_in_n_gram_scores(similarity_matrix[letter_index, :],
                                                                               n_gram_length, l)
            final_scores += self.position_matrix[n_gram_index, y_start_index + i] * n_gram_scores
        return np.min(final_scores)

    cdef np.ndarray[FLOAT64_t, ndim=1] transform_letter_scores_in_n_gram_scores(self, np.ndarray[FLOAT64_t, ndim=1]
                                                                                   letter_scores, int n_gram_length,
                                                                                   int index_in_n_gram):
        cdef np.ndarray[FLOAT64_t, ndim=1] n_gram_scores
        cdef int n_repeat, n_tile
        n_repeat = self.alphabet_length ** (n_gram_length - index_in_n_gram - 1)
        n_tile = self.alphabet_length ** index_in_n_gram
        n_gram_scores = np.tile(np.repeat(letter_scores, n_repeat), n_tile)
        return n_gram_scores


cdef class DnaMinBoundCalculator(MinBoundCalculator):
    def __init__(self, n_gram_length, alphabet_length, n_grams, n_gram_to_index, final_length, gs_kernel):
        self.n_gram_length = n_gram_length
        self.n_gram_to_index = n_gram_to_index
        super(DnaMinBoundCalculator, self).__init__(alphabet_length, n_grams, final_length, gs_kernel)

    cdef FLOAT64_t[::1] precompute_y_y_bound_for_each_length(self, int max_length):
        cdef FLOAT64_t[::1] y_y_bounds = np.zeros(max_length, dtype=np.float64)
        cdef FLOAT64_t length_bound
        cdef int length, j, k
        # For all possible prefix lengths
        for length in range(1, max_length):
            length_bound = 0
            # Position 1
            for j in range(length):
                # Position 2
                for k in range(j + 1, length):
                    # Positions are never the same, so they only contribute to the score if the distance is a factor
                    # of the alphabet length
                    if np.abs(j - k) % self.alphabet_length == 0:
                        length_bound += self.position_matrix[j, k]
            # No summation of n-gram length, so only length cases where j == k and S(.) = 1
            y_y_bounds[length] = length + 2 * length_bound
        return y_y_bounds

    cdef FLOAT64_t[::1] precompute_start_node_bounds(self, int final_length, list n_grams):
        cdef FLOAT64_t[::1] bounds = np.empty(len(n_grams), dtype=np.float64)
        cdef FLOAT64_t y_y_prime_bound
        cdef str n_gram
        cdef int i
        cdef int n_gram_start_index = final_length - self.n_gram_length
        for i, n_gram in enumerate(n_grams):
            y_y_prime_bound = self.compute_y_y_prime_bound(n_gram, n_gram_start_index)
            bounds[i] = self.start_node_real_values[i] + self.y_y_bounds[n_gram_start_index] + 2 * y_y_prime_bound
        return bounds

    cdef FLOAT64_t compute_y_y_prime_bound(self, str y, int y_start_index):
        # y is really y_prime in equation 15
        cdef np.ndarray[FLOAT64_t, ndim=2] similarity_matrix = np.asarray(self.similarity_matrix)
        cdef int i
        cdef FLOAT64_t y_y_prime_bound = 0
        # i in equation 15
        for i in range(y_start_index):
            y_y_prime_bound += self.compute_n_gram_y_y_prime_bound(i, y, y_start_index, similarity_matrix)
        return y_y_prime_bound


    cdef FLOAT64_t compute_n_gram_y_y_prime_bound(self, int n_gram_pos, str y, int y_start_index,
                                                  np.ndarray[FLOAT64_t, ndim=2] similarity_matrix):
        # Calculates the min part of equation 15. y is really y_prime in the equation.
        cdef int j, n_gram_index
        cdef str n_gram
        cdef np.ndarray[FLOAT64_t, ndim=1] final_scores
        final_scores = np.zeros(self.alphabet_length ** self.n_gram_length)
        # j in equation 15
        for j in range(len(y) - self.n_gram_length + 1):
            # For each position, compute the positional similarity and weight it by similarity of all n-grams to the
            # current n-gram
            n_gram = y[j:j+self.n_gram_length]
            n_gram_index = self.n_gram_to_index[n_gram]
            final_scores += self.position_matrix[n_gram_pos, y_start_index + j] * similarity_matrix[n_gram_index, :]
        return np.min(final_scores)


    cdef Bound compute_bound(self, str y, FLOAT64_t parent_real_value, int final_length):
        # y represents an n-gram at a given node of the DAG
        cdef FLOAT64_t gs_similarity = parent_real_value + self.gs_similarity_new_n_gram(y)
        cdef FLOAT64_t y_y_similarity = self.y_y_bounds[final_length - len(y) - self.n_gram_length + 1]
        cdef FLOAT64_t y_y_prime_similarity = self.compute_y_y_prime_bound(y, final_length - len(y) - self.n_gram_length + 1)
        cdef Bound bound
        bound.bound_value = gs_similarity + y_y_similarity + 2 * y_y_prime_similarity
        bound.real_value = gs_similarity
        return bound


    cdef FLOAT64_t gs_similarity_new_n_gram(self, str y):
        # The first n characters of y represent the newest n-gram. Compute the GS similarity of the newest n-gram to
        # the rest of the sequence.
        cdef int i, index_one, index_two
        cdef FLOAT64_t similarity = 0.0
        cdef str new_n_gram, other_n_gram
        new_n_gram = y[:self.n_gram_length]
        index_one = self.n_gram_to_index[new_n_gram]
        for i in range(1, len(y) - self.n_gram_length + 1):
            other_n_gram = y[i:i+self.n_gram_length]
            index_two = self.n_gram_to_index[other_n_gram]
            similarity += self.position_matrix[0, i] * self.similarity_matrix[index_one, index_two]
        # Multiply by 2 because this comparison should actually be done twice (once for each version of y) and add 1
        # because the new n-gram contributes exactly one case of self-similarity.
        return 1.0 + 2 * similarity


# For unit tests only
cdef class BoundCalculatorMock(BoundCalculator):
    cdef mock

    def __init__(self, mock):
        self.mock = mock

    cdef Bound compute_bound(self, str y, FLOAT64_t parent_real_value, int final_length):
        return self.mock.compute_bound(y, final_length)

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        return self.mock.get_start_node_real_values(final_length)

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        return self.mock.get_start_node_bounds(final_length)
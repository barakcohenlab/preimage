import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from cpython cimport bool


ctypedef np.float64_t FLOAT64_t
ctypedef np.int8_t INT8_t
ctypedef np.int64_t INT64_t


cdef inline INT64_t int_min(INT64_t a, INT64_t b): return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef element_wise_generic_string_kernel(INT8_t[:, ::1] X, INT64_t[::1] x_lengths, FLOAT64_t[:, ::1] position_matrix,
                                         INT64_t n_min, INT64_t n_max):
    cdef int i
    cdef FLOAT64_t[::1] kernel = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        kernel[i] = generic_string_non_blended_similarity(X[i], x_lengths[i], X[i], x_lengths[i], position_matrix,
                                                          n_min, n_max)
    return np.asarray(kernel)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline FLOAT64_t generic_string_non_blended_similarity(INT8_t[::1] x1, INT64_t x1_length, INT8_t[::1] x2,
                                                            INT64_t x2_length, FLOAT64_t[:,::1] position_matrix,
                                                            INT64_t n_min, INT64_t n_max):
    cdef int i,j,l, is_n_gram_equal
    cdef FLOAT64_t similarity = 0.
    for i in range(x1_length - n_max + 1):
        for j in range(x2_length - n_max + 1):
            is_n_gram_equal = 1
            for l in range(n_min - 1, n_max):
                is_n_gram_equal *= x1[i + l] == x2[j + l]
            similarity += position_matrix[i, j] * is_n_gram_equal
    return similarity
    

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generic_string_kernel_with_sigma_c(INT8_t[:,::1] X1, INT64_t[::1] x1_lengths, INT8_t[:,::1] X2,
                                         INT64_t[::1] x2_lengths, FLOAT64_t[:,::1] position_matrix,
                                         FLOAT64_t[:,::1] similarity_matrix, INT64_t n_min, INT64_t n_max,
                                         bool symmetric):
    cdef int i, j
    cdef FLOAT64_t[:,::1] gram_matrix = np.zeros((X1.shape[0], X2.shape[0]), dtype=np.float64)

    if symmetric:
        for i in range(X1.shape[0]):
            gram_matrix[i, i] = generic_string_kernel_similarity_with_sigma_c(X1[i], x1_lengths[i], X2[i],
                                                                              x2_lengths[i], position_matrix,
                                                                              similarity_matrix, n_min, n_max)
            for j in range(i):
                gram_matrix[i, j] = generic_string_kernel_similarity_with_sigma_c(X1[i], x1_lengths[i], X2[j],
                                                                                  x2_lengths[j], position_matrix,
                                                                                  similarity_matrix, n_min, n_max)
                gram_matrix[j, i] = gram_matrix[i, j]
    else:
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                gram_matrix[i, j] = generic_string_kernel_similarity_with_sigma_c(X1[i], x1_lengths[i], X2[j], 
                                                                                  x2_lengths[j], position_matrix, 
                                                                                  similarity_matrix, n_min, n_max)
    return np.asarray(gram_matrix)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline FLOAT64_t generic_string_kernel_similarity_with_sigma_c(INT8_t[::1] x1, INT64_t x1_length, INT8_t[::1] x2,
                                                                    INT64_t x2_length, FLOAT64_t[:,::1] position_matrix,
                                                                    FLOAT64_t[:,::1] similarity_matrix, INT64_t n_min,
                                                                    INT64_t n_max):
    cdef INT64_t i,j,l, max_length
    cdef FLOAT64_t similarity, current_similarity, n_gram_similarity
    similarity = 0.

    for i in range(x1_length):
        max_length = int_min(n_max, x1_length - i)
        for j in range(x2_length):
            if x2_length - j < max_length:
                max_length = x2_length - j
            current_similarity = 1.
            n_gram_similarity = 0.
            for l in range(n_min - 1, max_length):
                current_similarity *= similarity_matrix[x1[i+l], x2[j+l]]
                n_gram_similarity += current_similarity
            similarity += position_matrix[i,j] * n_gram_similarity
    return similarity


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef element_wise_generic_string_kernel_with_sigma_c(INT8_t[:,::1] X, INT64_t[::1] x_lengths,
                                                      FLOAT64_t[:,::1] position_matrix,
                                                      FLOAT64_t[:,::1] similarity_matrix,
                                                      INT64_t n_min, INT64_t n_max):
    cdef int i
    cdef FLOAT64_t[::1] kernel = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        kernel[i] = generic_string_kernel_similarity_with_sigma_c(X[i], x_lengths[i], X[i], x_lengths[i],
                                                                  position_matrix, similarity_matrix, n_min, n_max)
    return np.asarray(kernel)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generic_string_ngram_kernel_with_sigma_c(INT64_t[:, ::1] X1, INT64_t[::1] x1_lengths, INT64_t[:, ::1] X2,
                                               INT64_t[::1] x2_lengths, FLOAT64_t[:, ::1] position_matrix,
                                               FLOAT64_t[:, ::1] similarity_matrix, bool symmetric):
    """Implementation of the GS kernel when n_min == n_max and strings are pre-computed to n-grams with a sliding window."""
    cdef int i, j, id
    cdef FLOAT64_t[:, ::1] gram_matrix = np.zeros((X1.shape[0], X2.shape[0]), dtype=np.float64)

    if symmetric:
        for i in prange(X1.shape[0], nogil=True, schedule="static"):
            gram_matrix[i, i] = generic_string_ngram_kernel_similarity_with_sigma_c(X1[i], x1_lengths[i], X2[i],
                                                                                    x2_lengths[i], position_matrix,
                                                                                    similarity_matrix)
            for j in range(i):
                gram_matrix[i, j] = generic_string_ngram_kernel_similarity_with_sigma_c(X1[i], x1_lengths[i], X2[j],
                                                                                        x2_lengths[j], position_matrix,
                                                                                        similarity_matrix)
                gram_matrix[j, i] = gram_matrix[i, j]

    else:
        for i in prange(X1.shape[0], nogil=True, schedule="static"):
            for j in range(X2.shape[0]):
                gram_matrix[i, j] = generic_string_ngram_kernel_similarity_with_sigma_c(X1[i], x1_lengths[i], X2[j],
                                                                                        x2_lengths[j], position_matrix,
                                                                                        similarity_matrix)

    return np.asarray(gram_matrix)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline FLOAT64_t generic_string_ngram_kernel_similarity_with_sigma_c(INT64_t[::1] x1, INT64_t x1_length,
                                                                          INT64_t[::1] x2, INT64_t x2_length,
                                                                          FLOAT64_t[:, ::1] position_matrix,
                                                                          FLOAT64_t[:, ::1] similarity_matrix) nogil:
    """Implementation of GS kernel similarity for two strings when n_min == n_max and strings are pre-computed to n-grams."""
    cdef INT64_t i, j
    cdef FLOAT64_t similarity
    similarity = 0.0

    for i in range(x1_length):
        for j in range(x2_length):
            #FIXME currently guaranteed to be correct because for k-mer the similarity matrix is only non-zero if
            #x1[i] == x2[j], but need to make sure alphabet order is preserved for other sorts of matrices.
            similarity += position_matrix[i, j] * similarity_matrix[x1[i], x2[j]]

    return similarity


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef element_wise_generic_string_ngram_kernel_with_sigma_c(INT64_t[:, ::1] X, INT64_t[::1] x_lengths,
                                                            FLOAT64_t[:, ::1] position_matrix,
                                                            FLOAT64_t[:, ::1] similarity_matrix):
    """Implementation of GS kernel self-similarity when n_min == n_max and strings are pre-computed to n-grams."""
    cdef int i
    cdef FLOAT64_t[::1] kernel = np.empty(X.shape[0], dtype=np.float64)
    for i in prange(X.shape[0], nogil=True, schedule="static"):
        kernel[i] = generic_string_ngram_kernel_similarity_with_sigma_c(X[i], x_lengths[i], X[i], x_lengths[i],
                                                                        position_matrix, similarity_matrix)

    return np.asarray(kernel)

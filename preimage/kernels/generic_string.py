__author__ = 'amelie'

import numpy as np
from abc import ABC, abstractmethod

from preimage.datasets.loader import load_amino_acids_and_descriptors, load_dna_pentamers_and_shape_similarity
from preimage.kernels._generic_string import element_wise_generic_string_kernel, generic_string_kernel_with_sigma_c
from preimage.kernels._generic_string import element_wise_generic_string_kernel_with_sigma_c
from preimage.datasets.amino_acid_file import AminoAcidFile
from preimage.datasets.dna_shape_files import DnaShapeFiles
from preimage.utils.position import compute_position_weights_matrix
from preimage.utils.alphabet import transform_strings_to_integer_lists, transform_dna_to_pentamer_integer_lists


def element_wise_kernel(X, sigma_position, n, alphabet):
    """Compute the similarity of each string in X with itself in the Generic String kernel.

    Takes only in account the position penalties and the n-gram of length n. No n-gram penalties (no sigma_c).

    Parameters
    ----------
    X : array, shape = [n_samples]
        Strings, where n_samples is the number of examples in X.
    sigma_position : float
        Controls the penalty incurred when two n-grams are not sharing the same position.
    n : int
        N-gram length.
    alphabet : list
        List of letters.

    Returns
    -------
    kernel : array, shape = [n_samples]
        Similarity of each string with itself in the GS kernel, where n_samples is the number of examples in X.
        """
    X = np.array(X)
    x_lengths = np.array([len(x) for x in X], dtype=np.int64)
    max_length = np.max(x_lengths) - n + 1
    position_matrix = compute_position_weights_matrix(max_length, sigma_position)
    X_int = transform_strings_to_integer_lists(X, alphabet)
    kernel = element_wise_generic_string_kernel(X_int, x_lengths, position_matrix, n)
    return kernel


class BaseGenericStringKernel(ABC):
    """Abstract base class for generic string kernel.

    Contains attributes and methods that are common to all generic string kernels. The base sets up the
    infrastructure for computing the similarity between two strings by comparing each of their l-gram of length 1 to
    n. The comparison depends on the physical similarity of the respective l-grams and a shifting contribution term
    that decays exponentially rapidly with the distance between the starting position. Due to difference between DNA
    and protein, the physical similarity computation will differ and get implemented in derived classes.

    Attributes
    ----------
    properties_file_name : (abstract) string or dictionary
        Name of the file containing the amino acid substitution matrix or mapping of DNA shape similarity comparisons
        to the file name that has that comparison.
    sigma_position : float
        Controls the penalty incurred when two n-grams are not sharing the same position.
    sigma_physical : float
        Controls the penalty incurred when the encoding vectors of two sequences differ.
    n : int
        N-gram length.
    is_normalized : bool
        True if the kernel should be normalized, False otherwise.
    """
    @property
    @abstractmethod
    def properties_file_name(self):
        """An abstract attribute that must be implemented in derived classes. This attribute indicates the file name(s)
        for physical properties of the alphabet. Due to differences in DNA and amino acid, the usage of this
        attribute will vary.
        """
        raise NotImplementedError

    def __init__(self, sigma_position=1.0, sigma_physical=1.0, n=2, is_normalized=True):
        self.sigma_position = sigma_position
        self.sigma_physical = sigma_physical
        self.n = n
        self.is_normalized = is_normalized

    @abstractmethod
    def __call__(self, X1, X2):
        """An abstract method for calling the kernel.

        Parameters
        ----------
        X1 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X1.
        X2 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X2.

        """
        raise NotImplementedError

    def get_position_matrix(self, max_length):
        """Compute the position similarity weights

        Parameters
        ----------
        max_length : int
            Maximum position.

        Returns
        -------
        position_matrix : array, shape = [max_length, max_length]
            Similarity of each position with all the other positions.
        """
        position_matrix = compute_position_weights_matrix(max_length, self.sigma_position)
        return position_matrix

    def _get_lengths(self, X1, X2):
        """Get the length of every sequence in both datasets.

        Parameters
        ----------
        X1 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X1.
        X2 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X2.

        Returns
        -------
        max_length : int
            Length of the longest sequence in either X1 or X2
        x1_lengths : array, shape=[n_samples, ]
            ints representing the length of each sample in X1, where n_samples is the number of samples in X1.
        x2_lengths : array, shape=[n_samples, ]
            ints representing the length of each sample in X1, where n_samples is the number of samples in X2.

        """
        x1_lengths = np.array([len(x) for x in X1], dtype=np.int64)
        x2_lengths = np.array([len(x) for x in X2], dtype=np.int64)
        max_length = max(np.max(x1_lengths), np.max(x2_lengths))
        return max_length, x1_lengths, x2_lengths

    @abstractmethod
    def _normalize(self, gram_matrix, X1, x1_lengths, X2, x2_lengths, position_matrix, similarity_matrix, is_symmetric):
        """Normalize the Gram matrix (kernel) based on the self-similarity of X1 and X2.

        Parameters
        ----------
        gram_matrix : array, shape = [n_samples_x1, n_samples_x2]
            Similarity of each string of X1 with each string of X2, n_samples_x1 is the number of samples in X1 and
            n_samples_x2 is the number of samples in X2.
        X1 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X1.
        x1_lengths : array, shape=[n_samples, ]
            ints representing the length of each sample in X1, where n_samples is the number of samples in X1.
        X2 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X2.
        x2_lengths : array, shape=[n_samples, ]
            ints representing the length of each sample in X1, where n_samples is the number of samples in X2.
        position_matrix : array, shape = [max_length, max_length]
            Similarity of each position with all the other positions.
        similarity_matrix : array, shape = [n_alphabet, n_alphabet] OR list of such matrices
            Pairwise physical similarities of alphabet.
        is_symmetric : bool
            Indicates if the Gram matrix is symmetric or not.

        Returns
        -------
        gram_matrix : array, shape = [n_samples_x1, n_samples_x2]
            The normalized Gram matrix.

        """
        raise NotImplementedError

    @abstractmethod
    def element_wise_kernel(self, X):
        """Compute the similarity of each string of X with itself in the Generic String kernel.

        Parameters
        ----------
        X : array, shape = [n_samples]
            Strings, where n_samples is the number of examples in X.

        Returns
        -------
        kernel : array, shape = [n_samples]
            Similarity of each string with itself in the GS kernel, where n_samples is the number of examples in X.
        """
        raise NotImplementedError


class GenericStringKernel(BaseGenericStringKernel):
    """Generic String Kernel.

    Computes the similarity between two strings by comparing each of their l-gram of length 1 to n. Each l-gram
    comparison yields a score that depends on the similarity of their respective amino acids (letters) and a shifting
    contribution term that decays exponentially rapidly with the distance between the starting positions of the two
    substrings. The sigma_position parameter controls the shifting contribution term. The sigma_amino_acid parameter
    controls the amount of penalty incurred when the encoding vectors differ as measured by the squared Euclidean
    distance between these two vectors. The GS kernel outputs the sum of all the l-gram-comparison scores.

    Notes
    -----
    See http://graal.ift.ulaval.ca/bioinformatics/gs-kernel/ for the original code developed by Sebastien Giguere [1]_.

    References
    ----------
    .. [1] Sebastien Giguere, Mario Marchand, Francois Laviolette, Alexandre Drouin, and Jacques Corbeil. "Learning a
       peptide-protein binding affinity predictor with kernel ridge regression." BMC bioinformatics 14, no. 1 (2013):
       82.
    """
    @property
    def properties_file_name(self):
        return self._properties_file_name

    def __init__(self, amino_acid_file_name=AminoAcidFile.blosum62_natural, sigma_position=1.0, sigma_amino_acid=1.0,
                 n=2, is_normalized=True):
        super().__init__(sigma_position=sigma_position, sigma_physical=sigma_amino_acid, n=n,
                         is_normalized=is_normalized)
        self._properties_file_name = amino_acid_file_name
        self.alphabet, self.descriptors = self._load_amino_acids_and_normalized_descriptors()

    def __call__(self, X1, X2):
        """Compute the similarity of all the strings of X1 with all the strings of X2 in the Generic String Kernel.

        Parameters
        ----------
        X1 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X1.
        X2 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X2.

        Returns
        -------
        gram_matrix : array, shape = [n_samples_x1, n_samples_x2]
            Similarity of each string of X1 with each string of X2, n_samples_x1 is the number of samples in X1 and
            n_samples_x2 is the number of samples in X2.
        """
        X1 = np.array(X1)
        X2 = np.array(X2)
        amino_acid_similarity_matrix = self.get_alphabet_similarity_matrix()
        is_symmetric = bool(X1.shape == X2.shape and np.all(X1 == X2))
        max_length, x1_lengths, x2_lengths = self._get_lengths(X1, X2)
        position_matrix = self.get_position_matrix(max_length)
        X1_int = transform_strings_to_integer_lists(X1, self.alphabet)
        X2_int = transform_strings_to_integer_lists(X2, self.alphabet)
        gram_matrix = generic_string_kernel_with_sigma_c(X1_int, x1_lengths, X2_int, x2_lengths, position_matrix,
                                                         amino_acid_similarity_matrix, self.n, is_symmetric)
        gram_matrix = self._normalize(gram_matrix, X1_int, x1_lengths, X2_int, x2_lengths, position_matrix,
                                      amino_acid_similarity_matrix, is_symmetric)
        return gram_matrix

    def get_alphabet_similarity_matrix(self):
        """Compute the alphabet similarity weights

        Returns
        -------
        similarity_matrix : array, shape = [len(alphabet), len(alphabet)]
            Similarity of each amino acid (letter) with all the other amino acids.
        """
        distance_matrix = np.zeros((len(self.alphabet), len(self.alphabet)))
        np.fill_diagonal(distance_matrix, 0)
        for index_one, descriptor_one in enumerate(self.descriptors):
            for index_two, descriptor_two in enumerate(self.descriptors):
                distance = descriptor_one - descriptor_two
                squared_distance = np.dot(distance, distance)
                distance_matrix[index_one, index_two] = squared_distance
        distance_matrix /= 2. * (self.sigma_physical ** 2)
        return np.exp(-distance_matrix)

    def _load_amino_acids_and_normalized_descriptors(self):
        amino_acids, descriptors = load_amino_acids_and_descriptors(self.properties_file_name)
        normalization = np.array([np.dot(descriptor, descriptor) for descriptor in descriptors],
                                    dtype=np.float)
        normalization = normalization.reshape(-1, 1)
        descriptors /= np.sqrt(normalization)
        return amino_acids, descriptors

    def _normalize(self, gram_matrix, X1, x1_lengths, X2, x2_lengths, position_matrix, similarity_matrix, is_symmetric):
        if self.is_normalized:
            if is_symmetric:
                x1_norm = gram_matrix.diagonal()
                x2_norm = x1_norm
            else:
                x1_norm = element_wise_generic_string_kernel_with_sigma_c(X1, x1_lengths, position_matrix,
                                                                          similarity_matrix, self.n)
                x2_norm = element_wise_generic_string_kernel_with_sigma_c(X2, x2_lengths, position_matrix,
                                                                          similarity_matrix, self.n)
            gram_matrix = ((gram_matrix / np.sqrt(x2_norm)).T / np.sqrt(x1_norm)).T
        return gram_matrix

    def element_wise_kernel(self, X):
        """Compute the similarity of each string of X with itself in the Generic String kernel.

        Parameters
        ----------
        X : array, shape = [n_samples]
            Strings, where n_samples is the number of examples in X.

        Returns
        -------
        kernel : array, shape = [n_samples]
            Similarity of each string with itself in the GS kernel, where n_samples is the number of examples in X.
        """
        X = np.array(X)
        X_int = transform_strings_to_integer_lists(X, self.alphabet)
        x_lengths = np.array([len(x) for x in X], dtype=np.int64)
        max_length = np.max(x_lengths)
        similarity_matrix = self.get_alphabet_similarity_matrix()
        position_matrix = self.get_position_matrix(max_length)
        kernel = element_wise_generic_string_kernel_with_sigma_c(X_int, x_lengths, position_matrix, similarity_matrix,
                                                                 self.n)
        return kernel


    class GenericStringDnaKernel(BaseGenericStringKernel):
        """Generic String Kernel for DNA shape similarity.

        Computes the similarity between two strings of DNA sequence based on pre-computed similarity tables. Due to
        edge cases with base step parameters, multiple similarity tables are necessary. These tables are stored as a
        dictionary with the attribute properties_file_name. Most comparisons use the MidToMid table, but edge cases
        are necessary when using the left-most and right-most pentamer of a sequence.

        Note that self.n is 1 by default, rather than 5, because this is the size of an n-gram from the PENTAMER
        alphabet, not an n-gram from the DNA alphabet. Thus an n = 1 really corresponds to a DNA pentamer,
        n = 2 corresponds to a DNA hexamer, etc.
        """
        @property
        def properties_file_name(self):
            """Get a dictionary containing the DNA shape similarity tables.

            Returns
            -------
            _properties_file_name : dict, {str: str}
                Keys are the types of DNA shape comparisons, values are the names of the corresponding files.
            """
            return self._properties_file_name

        def __init__(self, shape_similarity_file_dict=DnaShapeFiles.dna_shape_core, sigma_position=1.0,
                     sigma_physical=1.0, n=1, is_normalized=True):
            super().__init__(sigma_position=sigma_position, sigma_physical=sigma_physical, n=n,
                             is_normalized=is_normalized)
            self._properties_file_name = shape_similarity_file_dict
            self.alphabet, self.similarity_tables_dict = load_dna_pentamers_and_shape_similarity(
                self.properties_file_name)

        def __call__(self, X1, X2):
            """Compute the similarity of all the strings of X1 with all the strings of X2.

            Parameters
            ----------
            X1 : array, shape=[n_samples, ]
                Strings, where n_samples is the number of samples in X1.
            X2 : array, shape=[n_samples, ]
                Strings, where n_samples is the number of samples in X2.

            Returns
            -------
            gram_matrix : array, shape = [n_samples_x1, n_samples_x2]
                Similarity of each string of X1 with each string of X2, n_samples_x1 is the number of samples in X1 and
                n_samples_x2 is the number of samples in X2.
            """
            X1 = np.array(X1)
            X2 = np.array(X2)
            is_symmetric = bool(X1.shape == X2.shape and np.all(X1 == X2))
            max_length, x1_lengths, x2_lengths = self._get_lengths(X1, X2)
            position_matrix = self.get_position_matrix(max_length)
            X1_int = transform_dna_to_pentamer_integer_lists(X1, self.alphabet)
            X2_int = transform_dna_to_pentamer_integer_lists(X2, self.alphabet)
            gram_matrix = generic_string_dna_kernel_with_sigma_c(X1_int, x1_lengths, X2_int, x2_lengths,
                                                                 position_matrix, self.similarity_tables_dict, self.n,
                                                                 is_symmetric)
            gram_matrix = self._normalize(gram_matrix, X1_int, x1_lengths, X2_int, x2_lengths, position_matrix,
                                          self.similarity_tables_dict, is_symmetric)
            return gram_matrix

        def _normalize(self, gram_matrix, X1, x1_lengths, X2, x2_lengths, position_matrix, similarity_tables,
                       is_symmetric):
            """Normalize the Gram matrix to the self-similarity of each string if indicated to do so by this kernel's attributes.

            Parameters
            ----------
            gram_matrix : array, shape = [n_samples_x1, n_samples_x2]
                Similarity of each string of X1 with each string of X2, n_samples_x1 is the number of samples in X1 and
                n_samples_x2 is the number of samples in X2.
            X1 : array, shape=[n_samples, ]
                Strings, where n_samples is the number of samples in X1.
            x1_lengths : array, shape=[n_samples, ]
            ints representing the length of each sample in X1, where n_samples is the number of samples in X1.
            X2 : array, shape=[n_samples, ]
                Strings, where n_samples is the number of samples in X2.
            x2_lengths : array, shape=[n_samples, ]
            ints representing the length of each sample in X2, where n_samples is the number of samples in X2.
            position_matrix : array, shape = [max_length, max_length]
            Similarity of each position with all the other positions.
            similarity_tables : dict, {str: np.array}
                Keys are the types of DNA shape comparisons, values are the corresponding similarity tables.
            is_symmetric : bool
                Indicates if the Gram matrix is symmetric.

            Returns
            -------
            gram_matrix : array, shape = [n_samples_x1, n_samples_x2]
                Normalized Gram matrix if indicated to do so by the kernel.
            """
            if self.is_normalized:
                if is_symmetric:
                    x1_norm = gram_matrix.diagonal()
                    x2_norm = x1_norm
                else:
                    x1_norm = element_wise_generic_string_dna_kernel_with_sigma_c(X1, x1_lengths, position_matrix,
                                                                                  similarity_tables, self.n)
                    x2_norm = element_wise_generic_string_dna_kernel_with_sigma_c(X2, x2_lengths, position_matrix,
                                                                                  similarity_tables, self.n)

                gram_matrix = ((gram_matrix / np.sqrt(x2_norm)).T / np.sqrt(x1_norm)).T

            return gram_matrix

        def element_wise_kernel(self, X):
            """Compute the similarity of each string of X with itself in the Generic String kernel.

            Parameters
            ----------
            X : array, shape = [n_samples]
                Strings, where n_samples is the number of examples in X.

            Returns
            -------
            kernel : array, shape = [n_samples]
                Similarity of each string with itself in the GS kernel, where n_samples is the number of examples in X.
            """
            X = np.array(X)
            X_int = transform_dna_to_pentamer_integer_lists(X, self.alphabet)
            x_lengths = np.array([len(x) for x in X], dtype=np.int64)
            max_length = np.max(x_lengths)
            position_matrix = self.get_position_matrix(max_length)
            kernel = element_wise_generic_string_dna_kernel_with_sigma_c(X_int, x_lengths, position_matrix,
                                                                         self.similarity_tables_dict, self.n)
            return kernel

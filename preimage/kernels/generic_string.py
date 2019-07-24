__author__ = 'amelie'

import numpy as np

from preimage.datasets.loader import load_amino_acids_and_descriptors, load_dna_pentamers_and_shape_similarity
from preimage.kernels._generic_string import element_wise_generic_string_kernel, generic_string_kernel_with_sigma_c
from preimage.kernels._generic_string import element_wise_generic_string_kernel_with_sigma_c
from preimage.datasets.amino_acid_file import AminoAcidFile
from preimage.datasets.dna_shape_files import DnaShapeFiles
from preimage.utils.position import compute_position_weights_matrix
from preimage.utils.alphabet import transform_strings_to_integer_lists, transform_dna_to_pentamer_integer_lists


def element_wise_kernel(X, sigma_position, n_min, n_max, alphabet):
    """Compute the similarity of each string in X with itself in the Generic String kernel.

    Takes only in account the position penalties and the n-gram of length n. No n-gram penalties (no sigma_c).

    Parameters
    ----------
    X : array, shape = [n_samples]
        Strings, where n_samples is the number of examples in X.
    sigma_position : float
        Controls the penalty incurred when two n-grams are not sharing the same position.
    n_min : int
        Min n-gram length
    n_max : int
        Max n-gram length.
    alphabet : list
        List of letters.

    Returns
    -------
    kernel : array, shape = [n_samples]
        Similarity of each string with itself in the GS kernel, where n_samples is the number of examples in X.
        """
    X = np.array(X)
    x_lengths = np.array([len(x) for x in X], dtype=np.int64)
    max_length = np.max(x_lengths) - n_max + 1
    position_matrix = compute_position_weights_matrix(max_length, sigma_position)
    X_int = transform_strings_to_integer_lists(X, alphabet)
    kernel = element_wise_generic_string_kernel(X_int, x_lengths, position_matrix, n_min, n_max)
    return kernel


class GenericStringKernel:
    """Generic String Kernel.

    Computes the similarity between two strings by comparing each of their l-gram of length n_min to n_max. Each l-gram
    comparison yields a score that depends on the similarity of their respective substrings and a shifting
    contribution term that decays exponentially rapidly with the distance between the starting positions of the two
    substrings. The sigma_position parameter controls the shifting contribution term. The sigma_properties parameter
    controls the amount of penalty incurred when the encoding vectors differ as measured by the squared Euclidean
    distance between these two vectors. The GS kernel outputs the sum of all the l-gram-comparison scores.

    Attributes
    ----------
    properties_file_name : string
        Name of the file containing the physical properties matrix.
    sigma_position : float
        Controls the penalty incurred when two n-grams are not sharing the same position.
    sigma_properties : float
        Controls the penalty incurred when the encoding vectors of two amino acids differ.
    n_min : int
        Minimum l-gram length.
    n_max : int
        Maximum l-gram length.
    is_normalized : bool
        True if the kernel should be normalized, False otherwise.

    Notes
    -----
    See http://graal.ift.ulaval.ca/bioinformatics/gs-kernel/ for the original code developed by Sebastien Giguere [1]_.

    References
    ----------
    .. [1] Sebastien Giguere, Mario Marchand, Francois Laviolette, Alexandre Drouin, and Jacques Corbeil. "Learning a
       peptide-protein binding affinity predictor with kernel ridge regression." BMC bioinformatics 14, no. 1 (2013):
       82.
    """
    def __init__(self, properties_file_name, sigma_position=1.0, sigma_properties=1.0, n_min=1, n_max=2,
                 is_normalized=True):
        """
        properties_file_name is a str and can be either the name of a file or one of the following shortcuts:
            "amino" is a shortcut for the BLOSUM62 matrix
            "dna_core" is a shortcut for the DNA shape similarity matrix based on the core 4 parameters
            "dna_full" is a shortcut for the DNA shape similarity matrix based on the full set of parameters
        """
        # Read in the properties file and get the appropriate alphabet
        if properties_file_name == "amino" or properties_file_name == AminoAcidFile.blosum62_natural:
            self.properties_file_name = AminoAcidFile.blosum62_natural
            self.alphabet, self.descriptors = self._load_amino_acids_and_normalized_descriptors()
        elif properties_file_name == "dna_core" or properties_file_name == DnaShapeFiles.dna_shape_core:
            self.properties_file_name = DnaShapeFiles.dna_shape_core
            # FIXME
            # self.alphabet, self.descriptors = pass
        elif properties_file_name == "dna_full" or properties_file_name == DnaShapeFiles.dna_shape_full:
            self.properties_file_name = DnaShapeFiles.dna_shape_full
            # FIXME
            # self.alphabet, self.descriptors = pass
        else:
            # FIXME error type
            raise ValueError("Did not recognize physical properties file.")

        self.sigma_position = sigma_position
        self.sigma_properties = sigma_properties
        self.n_min = n_min
        self.n_max = n_max
        self.is_normalized = is_normalized

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
                                                         amino_acid_similarity_matrix, self.n_min, self.n_max,
                                                         is_symmetric)
        gram_matrix = self._normalize(gram_matrix, X1_int, x1_lengths, X2_int, x2_lengths, position_matrix,
                                      amino_acid_similarity_matrix, is_symmetric)
        return gram_matrix

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
        distance_matrix /= 2. * (self.sigma_properties ** 2)
        return np.exp(-distance_matrix)

    def _load_amino_acids_and_normalized_descriptors(self):
        amino_acids, descriptors = load_amino_acids_and_descriptors(self.properties_file_name)
        normalization = np.array([np.dot(descriptor, descriptor) for descriptor in descriptors],
                                    dtype=np.float)
        normalization = normalization.reshape(-1, 1)
        descriptors /= np.sqrt(normalization)
        return amino_acids, descriptors

    def _get_lengths(self, X1, X2):
        x1_lengths = np.array([len(x) for x in X1], dtype=np.int64)
        x2_lengths = np.array([len(x) for x in X2], dtype=np.int64)
        max_length = max(np.max(x1_lengths), np.max(x2_lengths))
        return max_length, x1_lengths, x2_lengths

    def _normalize(self, gram_matrix, X1, x1_lengths, X2, x2_lengths, position_matrix, similarity_matrix, is_symmetric):
        if self.is_normalized:
            if is_symmetric:
                x1_norm = gram_matrix.diagonal()
                x2_norm = x1_norm
            else:
                x1_norm = element_wise_generic_string_kernel_with_sigma_c(X1, x1_lengths, position_matrix,
                                                                          similarity_matrix, self.n_min, self.n_max)
                x2_norm = element_wise_generic_string_kernel_with_sigma_c(X2, x2_lengths, position_matrix,
                                                                          similarity_matrix, self.n_min, self.n_max)
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
                                                                 self.n_min, self.n_max)
        return kernel
__author__ = 'amelie, rfriedman22'

import numpy as np

from preimage.datasets.loader import load_amino_acids_and_descriptors, load_dna_pentamers_and_shape_similarity
from preimage.kernels._generic_string import element_wise_generic_string_kernel, generic_string_kernel_with_sigma_c, \
    element_wise_generic_string_kernel_with_sigma_c, generic_string_ngram_kernel_with_sigma_c, \
    element_wise_generic_string_ngram_kernel_with_sigma_c
from preimage.datasets.amino_acid_file import AminoAcidFile
from preimage.datasets.dna_shape_files import DnaShapeFiles
from preimage.utils.position import compute_position_weights_matrix
from preimage.utils.alphabet import transform_strings_to_integer_lists, transform_dna_to_ngram_integer_lists, \
    unique_dna_n_gram


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
        Name of the file containing the physical properties matrix or identifier of the kernel type (e.g. dna_kmer).
    sigma_position : float
        Controls the penalty incurred when two n-grams are not sharing the same position.
    sigma_properties : float
        Controls the penalty incurred when the encoding vectors of two alphabet members differ.
    n_min : int
        Minimum l-gram length.
    n_max : int
        Maximum l-gram length.
    is_normalized : bool
        True if the kernel should be normalized, False otherwise.
    alphabet : list-like
        Unique members of the alphabet, each item is a string.
    distance_matrix : matrix-like
        Squared Euclidean distance between the properties of every alphabet member with every other alphabet member.

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
            "dna_kmer" indicates a k-mer (spectral) kernel for DNA. In this case, n_min must equal n_max,
            alphabet represents all non-redundant k-mers (reverse compliments are not counted), and distance_matrix
            is an identity matrix.
            "dna_core" is a shortcut for the DNA shape similarity matrix based on the core 4 parameters
            "dna_full" is a shortcut for the DNA shape similarity matrix based on the full set of parameters
        """
        # Read in the properties file and get the appropriate alphabet
        if properties_file_name == "amino" or properties_file_name == AminoAcidFile.blosum62_natural:
            self.properties_file_name = AminoAcidFile.blosum62_natural
            self.alphabet, self.distance_matrix = self._load_amino_acids_and_normalized_descriptors()

        elif properties_file_name == "dna_kmer":
            # Make sure n_min == n_max
            if n_min != n_max:
                raise ValueError("Specified a DNA k-mer kernel, but n_min does not equal n_max.")

            # Make sure sigma_properties is 0
            if sigma_properties != 0:
                raise ValueError(f"DNA k-mer kernel requires sigma_properties = 0, but {sigma_properties} was "
                                 f"specified.")

            self.properties_file_name = properties_file_name
            self.alphabet = unique_dna_n_gram(n_min)
            # Set the distance matrix to the identity matrix
            self.distance_matrix = np.eye(len(self.alphabet))

        elif properties_file_name == "dna_core" or properties_file_name == DnaShapeFiles.dna_shape_core:
            self.properties_file_name = DnaShapeFiles.dna_shape_core
            raise NotImplementedError("Core DNA shape not implemented yet")
            # FIXME
            # self.alphabet, self.distance_matrix = pass

        elif properties_file_name == "dna_full" or properties_file_name == DnaShapeFiles.dna_shape_full:
            self.properties_file_name = DnaShapeFiles.dna_shape_full
            raise NotImplementedError("Full DNA shape not implemented yet")
            # FIXME
            # self.alphabet, self.distance_matrix = pass

        else:
            raise ValueError("Did not recognize physical properties name.")

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
        alphabet_similarity_matrix = self.get_alphabet_similarity_matrix()
        is_symmetric = bool(X1.shape == X2.shape and np.all(X1 == X2))
        max_length, x1_lengths, x2_lengths = self._get_lengths(X1, X2)
        position_matrix = self.get_position_matrix(max_length)

        # Transform sequences to ints and get the appropriate function for computing the kernel depending on what was
        #  specified
        if self.properties_file_name == AminoAcidFile.blosum62_natural:
            transform = lambda x: transform_strings_to_integer_lists(x, self.alphabet)
            c_fun = lambda x1, x2: generic_string_kernel_with_sigma_c(x1, x1_lengths, x2, x2_lengths,
                                                                      position_matrix, alphabet_similarity_matrix,
                                                                      self.n_min, self.n_max, is_symmetric)
            c_norm_fun = lambda x, x_len: element_wise_generic_string_kernel_with_sigma_c(
                x, x_len, position_matrix, alphabet_similarity_matrix, self.n_min, self.n_max
            )
        elif self.properties_file_name == "dna_kmer":
            transform = lambda x: transform_dna_to_ngram_integer_lists(x, self.alphabet, self.n_min)
            c_fun = lambda x1, x2: generic_string_ngram_kernel_with_sigma_c(x1, x1_lengths, x2, x2_lengths,
                                                                            position_matrix,
                                                                            alphabet_similarity_matrix, is_symmetric)
            c_norm_fun = lambda x, x_len: element_wise_generic_string_ngram_kernel_with_sigma_c(
                x, x_len, position_matrix, alphabet_similarity_matrix
            )
        else:
            raise NotImplementedError("String to int transformation not implemented for this kernel!")

        X1_int = transform(X1)
        X2_int = transform(X2)
        gram_matrix = c_fun(X1_int, X2_int)
        gram_matrix = self._normalize(gram_matrix, X1_int, x1_lengths, X2_int, x2_lengths, is_symmetric, c_norm_fun)
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
            Similarity of each letter with all the other letters.
        """
        # If sigma_properties is 0, matrix reduces to identity matrix
        if self.sigma_properties == 0:
            similarity_matrix = np.eye(len(self.alphabet))
        else:
            similarity_matrix = self.distance_matrix
            similarity_matrix /= 2. * (self.sigma_properties ** 2)
            similarity_matrix = np.exp(-similarity_matrix)

        return similarity_matrix

    def _load_amino_acids_and_normalized_descriptors(self):
        """Normalize the amino acid descriptors and then compute the squared Euclidean distance between them."""
        amino_acids, descriptors = load_amino_acids_and_descriptors(self.properties_file_name)
        normalization = np.array([np.dot(descriptor, descriptor) for descriptor in descriptors],
                                    dtype=np.float)
        normalization = normalization.reshape(-1, 1)
        descriptors /= np.sqrt(normalization)

        # Now that the descriptors are normalized, compute the pairwise squared Euclidean distance between every
        # amino acid pair.
        distance_matrix = np.zeros((len(amino_acids), len(amino_acids)))
        for index_one, descriptor_one in enumerate(descriptors):
            for index_two, descriptor_two in enumerate(descriptors):
                distance = descriptor_one - descriptor_two
                squared_distance = np.dot(distance, distance)
                distance_matrix[index_one, index_two] = squared_distance

        return amino_acids, distance_matrix

    def _get_lengths(self, X1, X2):
        x1_lengths = np.array([len(x) - self.n_min + 1 for x in X1], dtype=np.int64)
        x2_lengths = np.array([len(x) - self.n_min + 1 for x in X2], dtype=np.int64)
        max_length = max(np.max(x1_lengths), np.max(x2_lengths))
        return max_length, x1_lengths, x2_lengths

    def _normalize(self, gram_matrix, X1, x1_lengths, X2, x2_lengths, is_symmetric, c_norm_fun):
        if self.is_normalized:
            if is_symmetric:
                x1_norm = gram_matrix.diagonal()
                x2_norm = x1_norm
            else:
                x1_norm = c_norm_fun(X1, x1_lengths)
                x2_norm = c_norm_fun(X2, x2_lengths)
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
__author__ = 'amelie, rfriedman22'

import numpy as np

from preimage.datasets.loader import load_amino_acids_and_descriptors, load_dna_pentamers_and_shape_similarity
from preimage.kernels._generic_string import element_wise_generic_string_kernel, generic_string_kernel_with_sigma_c, \
    element_wise_generic_string_kernel_with_sigma_c, generic_string_ngram_kernel_with_sigma_c, \
    element_wise_generic_string_ngram_kernel_with_sigma_c
from preimage.datasets.amino_acid_file import AminoAcidFile
from preimage.datasets.dna_shape_files import DnaShapeFiles
from preimage.utils.position import compute_position_weights_matrix
from preimage.utils.alphabet import get_n_grams, reverse_compliment, transform_strings_to_integer_lists, \
    transform_dna_to_ngram_integer_lists, unique_dna_n_gram, Alphabet


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
            self.alphabet = get_n_grams(Alphabet.dna, n_max)
            rc_dict = {i: reverse_compliment(i) for i in self.alphabet}
            # Set the distance matrix to the identity matrix, then fill in any off-diagonals for reverse compliments
            distance_matrix = np.eye(len(self.alphabet))
            for row, ngram_one in enumerate(self.alphabet):
                for col, ngram_two in enumerate(self.alphabet):
                    if ngram_one == rc_dict[ngram_two]:
                        distance_matrix[row, col] += 1

            self.distance_matrix = distance_matrix

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
        X1_int = self.transform(X1)
        X2_int = self.transform(X2)
        max_length, x1_lengths, x2_lengths = self._get_lengths(X1, X2)
        position_matrix = self.get_position_matrix(max_length)

        # Get the appropriate function for computing the kernel depending on what was specified
        if self.is_amino_acid():
            c_fun = lambda x1, x2: generic_string_kernel_with_sigma_c(x1, x1_lengths, x2, x2_lengths,
                                                                      position_matrix, alphabet_similarity_matrix,
                                                                      self.n_min, self.n_max, is_symmetric)
            c_norm_fun = lambda x, x_len: element_wise_generic_string_kernel_with_sigma_c(
                x, x_len, position_matrix, alphabet_similarity_matrix, self.n_min, self.n_max
            )
        elif self.is_dna_kmer():
            c_fun = lambda x1, x2: generic_string_ngram_kernel_with_sigma_c(x1, x1_lengths, x2, x2_lengths,
                                                                            position_matrix,
                                                                            alphabet_similarity_matrix, is_symmetric)
            c_norm_fun = lambda x, x_len: element_wise_generic_string_ngram_kernel_with_sigma_c(
                x, x_len, position_matrix, alphabet_similarity_matrix
            )
        else:
            raise NotImplementedError("Kernel computation not implemented for this version!")

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
        # If DNA k-mers, then similarity matrix is just the precomputed distance matrix
        if self.is_dna_kmer():
            similarity_matrix = self.distance_matrix
        # If sigma_properties is 0, matrix reduces to identity matrix
        elif self.sigma_properties == 0:
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

    def transform(self, x):
        """Transform strings to int representations.

        Parameters
        ----------
        x : list-like
            Strings of the input sequences.

        Returns
        -------
        x_int : 2D ndarray
            Each input sequence is represented as an ndarray of ints.
        """
        if self.is_amino_acid():
            x_int = transform_strings_to_integer_lists(x, self.alphabet)
        elif self.is_dna_kmer():
            x_int = transform_dna_to_ngram_integer_lists(x, self.alphabet, self.n_max)
        else:
            raise NotImplementedError("Transformation not implemented for this kernel.")
        return x_int

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
        X_int = self.transform(X)
        x_lengths = np.array([len(x) - self.n_min + 1 for x in X], dtype=np.int64)
        max_length = np.max(x_lengths)
        similarity_matrix = self.get_alphabet_similarity_matrix()
        position_matrix = self.get_position_matrix(max_length)
        if self.is_amino_acid():
            kernel = element_wise_generic_string_kernel_with_sigma_c(X_int, x_lengths, position_matrix, similarity_matrix,
                                                                     self.n_min, self.n_max)
        elif self.is_dna_kmer():
            kernel = element_wise_generic_string_ngram_kernel_with_sigma_c(X_int, x_lengths, position_matrix,
                                                                           similarity_matrix)
        else:
            raise NotImplementedError("Element-wise kernel not implemented for this kernel!")

        return kernel

    def save_gram_lower_triangle(self, gram_matrix, filename, delim="\t"):
        """Save the lower triangle of the Gram matrix to file. Assumes the matrix is symmetric.

        Parameters
        ----------
        gram_matrix : matrix-like
            The Gram matrix computed by this kernel.
        filename : str
            Name of the file to write.
        delim : str
            Delimiter in the file

        Returns
        -------
        self
        """
        if gram_matrix.shape[0] != gram_matrix.shape[1]:
            raise ValueError("Gram matrix is not symmetric, cannot take lower triangle.")

        out_str = ""
        for row in range(gram_matrix.shape[0]):
            out_str += delim.join([f"{i:1.6e}" for i in gram_matrix[row, :row + 1]]) + "\n"
        with open(filename, "w") as fout:
            fout.write(out_str)
        return self

    def set_gram_matrix_from_file(self, filename, delim="\t", lower_triangle=True):
        """Read in a precomputed Gram matrix from a file and return the matrix.

        Parameters
        ----------
        filename : str
            Name of the file with the precomputed Gram matrix.
        delim : str
            Delimiter in the file.
        lower_triangle : bool
            If True, the file contains the lower triangle of the Gram matrix. Otherwise, assumes the file is a full
            matrix.

        Returns
        -------
        gram_matrix : matrix-like
            Dense representation of the Gram matrix.
        """
        if lower_triangle:
            # First get the number of lines
            n_lines = 0
            with open(filename) as fin:
                for line in fin:
                    n_lines += 1

            # Initialize the Gram matrix as the identity matrix. Since it is symmetric, the diagonal must be ones.
            gram_matrix = np.eye(n_lines)
            with open(filename) as fin:
                for row, line in enumerate(fin):
                    line = map(float, line.split(delim))
                    for col, val in enumerate(line):
                        if row != col:
                            gram_matrix[row, col] = val
                            gram_matrix[col, row] = val

        else:
            gram_matrix = np.genfromtxt(filename, delimiter=delim)

        return gram_matrix

    def get_sub_matrix(self, row_idx, col_idx, gram_matrix):
        """Get a sub-Gram matrix. If row_idx is not the same as col_idx, then the rows should represent new data (
        i.e. test data) while columns represent the original, known training data. Raises an Error if the Gram matrix
        is not already computed.

        Parameters
        ----------
        row_idx : list-like
            Indices of the rows to extract from the Gram matrix.
        col_idx : list-like
            Indices of the columns to extract from the Gram matrix.
        gram_matrix : matrix-like
            Dense representation of the Gram matrix.

        Returns
        -------
        sub_mat : 2D ndarray
            The contents of the sub-Gram matrix
        """
        sub_mat = gram_matrix[np.ix_(row_idx, col_idx)]
        return sub_mat

    def is_dna_kmer(self):
        return self.properties_file_name == "dna_kmer"

    def is_amino_acid(self):
        return self.properties_file_name == AminoAcidFile.blosum62_natural

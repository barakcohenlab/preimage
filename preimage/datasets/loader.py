"""Loader for datasets and amino acids"""

__author__ = "amelie"

import pickle
import gzip
import os

import numpy as np

from preimage.datasets.amino_acid_file import AminoAcidFile
from preimage.datasets.dna_shape_files import DnaShapeFiles


class StructuredOutputDataset:
    """Structured output dataset.

    Attributes
    ----------
    X : array, shape = [n_samples, n_features]
        Vectors, where n_samples is the number of samples and n_features is the number of features.
    Y : array, shape = [n_samples, ]
        Target strings.
    y_lengths : array, shape = [n_samples]
        Length of each string in Y.
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.y_lengths = np.array([len(y) for y in Y])


class StandardDataset:
    """Classification or Regression Dataset.

    Attributes
    ----------
    X : array, shape = [n_samples, ]
        Strings, where n_samples is the number of samples.
    y : array, shape = [n_samples]
        Target values.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y


def load_ocr_letters(fold_id=0):
    """Load the OCR letter dataset.

    This dataset consists of word images with their corresponding word output.
    The original OCR letter dataset can be downloaded here : http://ai.stanford.edu/~btaskar/ocr/.

    Parameters
    ----------
    fold_id : int
        Id (0-9) of the fold used for training. The remaining examples are used for testing.

    Returns
    -------
    train_dataset : StructuredOutputDataset
        Training dataset.
    test_dataset : StructuredOutputDataset
        Testing dataset.
    """
    data = __load_gz_pickle_file("ocrletters.pickle.gz")
    train_indexes = np.where(data["fold_ids"] == fold_id)
    test_indexes = np.where(data["fold_ids"] != fold_id)
    Y_train = np.array(data["y"], dtype=np.str)[train_indexes]
    Y_test = np.array(data["y"], dtype=np.str)[test_indexes]
    train_dataset = StructuredOutputDataset(data["X"][train_indexes], Y_train)
    test_dataset = StructuredOutputDataset(data["X"][test_indexes], Y_test)
    return train_dataset, test_dataset


def __load_gz_pickle_file(file_name):
    module_path = os.path.dirname(__file__)
    gzip_reader = gzip.open(os.path.join(module_path, file_name), "rb")
    data = pickle.loads(gzip_reader.read())
    return data


def load_camps_dataset():
    """Load the CAMPs dataset consisting of 101 cationic antimicrobial pentadecapeptides.

    Returns
    -------
    train_dataset: StandardDataset
        Training dataset.
    """
    return __load_peptide_dataset("camps.pickle")


def load_bpps_dataset():
    """Load the BPPs dataset consisting of 31 bradykinin-potentiating pentapeptides.

    Returns
    -------
    train_dataset: StandardDataset
        Training dataset.
    """
    return __load_peptide_dataset("bpps.pickle")


def __load_peptide_dataset(file_name):
    data = __load_pickle_file(file_name)
    X = np.array(data["X"], dtype=np.str)
    train_dataset = StandardDataset(X, data["y"])
    return train_dataset


def __load_pickle_file(file_name):
    module_path = os.path.dirname(__file__)
    data_file = open(os.path.join(module_path, file_name), "rb")
    data = pickle.load(data_file)
    return data


def load_amino_acids_and_descriptors(file_name=AminoAcidFile.blosum62_natural):
    """Load amino acids and descriptors

    Parameters
    ----------
    file_name : string
        file name of the amino acid matrix.

    Returns
    -------
    amino_acids: list
        A list of amino acids (letters).
    descriptors: array, shape = [n_amino_acids, n_amino_acids]
        Substitution cost of each amino acid with all the other amino acids, where n_amino_acids is the number of
        amino acids.
    """
    path_to_file = os.path.join(os.path.dirname(__file__), "amino_acid_matrix", file_name)
    with open(path_to_file, "r") as data_file:
        lines = data_file.readlines()
    splitted_lines = np.array([line.split() for line in lines])
    amino_acids = [str(letter) for letter in splitted_lines[:, 0]]
    descriptors = np.array(splitted_lines[:, 1:], dtype=np.float)
    return amino_acids, descriptors


def load_dna_pentamers_and_shape_similarity(file_names=DnaShapeFiles.dna_shape_core, sigma_physical=1.0):
    """Load DNA pentamers and the 6 DNA shape lookup tables, then normalize and exponentiate the matrix.

    Parameters
    ----------
    file_names : dict, {str: str}
        File names for the 6 DNA shape similarity matrices. Keys are types of comparisons, values are the filenames.
    sigma_physical : float
        Normalization hyperparameter for physicochemical properties. After loading the distance matrix, normalize

    Returns
    -------
    pentamers: list
        A list of DNA pentamers (strings). Reverse compliments are excluded, e.g. if AAAAA is present then TTTTT is not.
    similarity_tables: dict, {str: array, shape = (n_pentamers, n_pentamers)}
        DNA shape similarity table of each pentamer with all the other pentamers, where n_pentamers is the number of
        DNA pentamers, normalized to the sigma hyperparameter, and then exponentiated. There is one table for each
        type of comparison (e.g. L-to-L, L-to-M). Keys are the comparison type, values are the similarity tables.
    """
    path_to_files = os.path.join(os.path.dirname(__file__), "dna_shape_matrix")
    similarity_tables = {}
    for comparison, file in file_names.items():
        with open(os.path.join(path_to_files, file), "r") as data_file:
            lines = data_file.readlines()
        splitted_lines = np.array([line.split() for line in lines])
        pentamers = [str(pentamer) for pentamer in splitted_lines[:, 0]]
        similarity = np.array(splitted_lines[:, 1:], dtype=np.float)
        similarity /= (2.0 * (sigma_physical ** 2))
        similarity = np.exp(-similarity)
        similarity_tables[comparison] = similarity

    return pentamers, similarity_tables

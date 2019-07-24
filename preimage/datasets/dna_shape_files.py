__author__ = "Ryan Z. Friedman"


class DnaShapeFiles:
    """File names of DNA shape similarity matrices.

    Attributes
    ----------
    dna_shape_core : str
        File names for the similarity matrix for the 4 core DNA shape parameters: MGW, ProT, HelT, Roll.
    dna_shape_full : str
        File names for the similarity matrix for the full set of 14 DNA shape parameters.
    """
    dna_shape_core = "dna_shape_similarity_core.txt"
    dna_shape_full = "dna_shape_similarity_full.txt"
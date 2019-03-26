__author__ = "Ryan Z. Friedman"


def generate_all_six_files(mat_type):
    """Generate the file name for all 6 DNA shape similarity matrices.

    Parameters
    ----------
    mat_type : string
        The type of shape similarity matrices to load, should be the end of the filename before .txt.

    Returns
    -------
    similarity_files : dict, {str: str}
        Keys are types of shape comparisons, values are the filenames.

    """
    prefix = "dnaShapeSimilarity"
    similarity_keys = ["LeftToLeft", "LefttoMid", "LeftToRight", "MidToMid", "MidToRight", "RightToRight"]
    similarity_files = {i: f"{prefix}{i}{mat_type}.txt" for i in similarity_keys}
    return similarity_files


class DnaShapeFiles:
    """File names of DNA shape similarity matrices. For each set of DNA shape features, there are six similarity
    matrices to handle all the edge cases.

    Attributes
    ----------
    dna_shape_core : dict, {str: str}
        File names for the 6 similarity matrices for the 4 core DNA shape parameters: MGW, ProT, HelT, Roll.
        Keys are types of shape comparisons, values are the filenames.
    dna_shape_full : dict, {str: str}
        File names for the 6 similarity matrices for the full set of 14 DNA shape parameters.
        Keys are types of shape comparisons, values are the filenames.
    """
    dna_shape_core = generate_all_six_files("Core")
    dna_shape_full = generate_all_six_files("Full")
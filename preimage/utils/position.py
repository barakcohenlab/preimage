__author__ = 'amelie, rfriedman22'

import numpy as np


def compute_position_weights(position_index, max_position, sigma_position):
    position_penalties = np.array([(position_index - j) ** 2 for j in range(max_position)], dtype=np.float)
    position_penalties /= -2. * (sigma_position ** 2)
    return np.exp(position_penalties)


def compute_position_weights_matrix(max_position, sigma_position):
    # If sigma_position is 0, this becomes the identity matrix
    if sigma_position == 0:
        return np.eye(max_position)

    positions = np.arange(max_position, dtype=np.float)
    position_penalties = np.subtract.outer(positions, positions)
    position_penalties = np.square(position_penalties) / (-2 * (sigma_position ** 2))
    return np.exp(position_penalties)
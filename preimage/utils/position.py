__author__ = 'amelie'

import numpy as np


def compute_position_weights(position_index, max_position, sigma_position):
    position_penalties = np.array([(position_index - j) ** 2 for j in range(max_position)], dtype=np.float)
    position_penalties /= -2. * (sigma_position ** 2)
    return np.exp(position_penalties)


def compute_position_weights_matrix(max_position, sigma_position):
    position_penalties = np.array([(i - j) for i in range(max_position) for j in range(max_position)],
                                     dtype=np.float)
    position_penalties = position_penalties.reshape(max_position, max_position)
    position_penalties = np.square(position_penalties) / (-2 * (sigma_position ** 2))
    return np.exp(position_penalties)
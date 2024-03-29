__author__ = 'amelie, rfriedman22'

from sklearn.base import BaseEstimator

from preimage.inference.graph_builder import GraphBuilder
from preimage.inference.branch_and_bound import branch_and_bound_multiple_solutions
from preimage.inference.bound_factory import get_gs_similarity_node_creator
from preimage.features.gs_similarity_feature_space import GenericStringSimilarityFeatureSpace


class StringMaximizationModel(BaseEstimator):
    # max_time is in seconds
    def __init__(self, alphabet, n_max, gs_kernel, max_time, n_min=1):
        self._n_max = int(n_max)
        self._n_min = int(n_min)
        self._alphabet = alphabet
        self._graph_builder = GraphBuilder(self._alphabet, self._n_max)
        self._gs_kernel = gs_kernel
        self._max_time = max_time
        self._is_normalized = True
        self._node_creator_ = None
        self._y_length_ = None

    def fit(self, X, learned_weights, y_length):
        feature_space = GenericStringSimilarityFeatureSpace(self._alphabet, self._n_max, X, self._is_normalized,
                                                            self._gs_kernel, n_min=self._n_min)
        gs_weights = feature_space.compute_weights(learned_weights, y_length)
        graph = self._graph_builder.build_graph(gs_weights, y_length)
        self._node_creator_ = get_gs_similarity_node_creator(self._alphabet, self._n_max, graph, gs_weights, y_length,
                                                             self._gs_kernel)
        self._y_length_ = y_length
        return graph

    def predict(self, n_predictions):
        strings, bounds = branch_and_bound_multiple_solutions(self._node_creator_, self._y_length_, n_predictions,
                                                              self._alphabet, self._max_time)
        return strings, bounds

__author__ = 'amelie'

from preimage.datasets.loader import load_ocr_letters
from preimage.learners.structured_krr import StructuredKernelRidgeRegression
from preimage.kernels.polynomial import PolynomialKernel
from preimage.models.weighted_degree_model import WeightedDegreeModel
from preimage.utils.alphabet import Alphabet
from preimage.metrics.structured_output import zero_one_loss, hamming_loss, levenshtein_loss


if __name__ == '__main__':
    # You should find best parameters with cross-validation
    poly_kernel = PolynomialKernel(degree=2)
    alpha = 1e-5
    n_predictions = 50
    n = 3
    is_using_length = True

    print('Weighted degree model on OCR Letter Dataset')
    train_dataset, test_dataset = load_ocr_letters(fold_id=0)
    inference_model = WeightedDegreeModel(Alphabet.latin, n, is_using_length)
    learner = StructuredKernelRidgeRegression(alpha, poly_kernel, inference_model)

    print('\ntraining ...')
    learner.fit(train_dataset.X, train_dataset.Y, train_dataset.y_lengths)

    print('predict ...')
    Y_predictions = learner.predict(test_dataset.X[0:n_predictions], test_dataset.y_lengths[0:n_predictions])

    print('\nY predictions: ', Y_predictions)
    print('Y real: ', test_dataset.Y[0:n_predictions])

    print('\nResults:')
    print('zero_one_loss', zero_one_loss(test_dataset.Y[0:n_predictions], Y_predictions[0:n_predictions]))
    print('levenshtein_loss', levenshtein_loss(test_dataset.Y[0:n_predictions], Y_predictions[0:n_predictions]))
    if is_using_length:
        print('hamming_loss', hamming_loss(test_dataset.Y[0:n_predictions], Y_predictions[0:n_predictions]))
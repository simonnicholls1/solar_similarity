from services.analysis.analysis import Analysis
import numpy as np
import pytest

def test_pair_most_like_given_ok():
    analysis_service = Analysis()
    a = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 1, 0]])
    labels = ['one', 'two', 'three']
    results = analysis_service.pair_most_like_given(a, 'three', labels, True)
    assert results == 'two'

def test_empty_score_matrix_error():
    analysis_service = Analysis()
    a = np.array([])
    labels = ['one', 'two', 'three']
    with pytest.raises(ValueError) as err:
        results = analysis_service.pair_most_like_given(a, 'three', labels, True)
    assert str(err.value) == 'Score matrix is empty!'




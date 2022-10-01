import pytest
from services.similarity.cosine_similarity import CosineSimilarity
from services.similarity.euclidean_similarity import EuclideanSimilarity
import pandas as pd
import numpy as np


def test_cosine_similarity_ok():
    cs = CosineSimilarity()
    a = np.array([2, 1, 2, 3, 2, 9])
    b = np.array([3, 4, 2, 4, 5, 5])
    similarity = cs.calculate_similarity(a, b)
    assert round(similarity, 3) == 0.819

def test_cosine_similarity_matrix_ok():
    cs = CosineSimilarity()
    a = pd.DataFrame([[2, 1, 2, 3, 2, 9], [3, 4, 2, 4, 5, 5]])
    similarity = cs.similarity(a)
    assert round(similarity[0, 0], 3) == 1.000
    assert round(similarity[1, 0], 3) == 0.819
    assert round(similarity[0, 1], 3) == 0.000
    assert round(similarity[1, 1], 3) == 1.000

def test_euclidean_similarity_ok():
    es = EuclideanSimilarity()
    a = pd.DataFrame([[2, 6, 7, 7, 5, 13, 14, 17, 11, 8], [3, 5, 5, 3, 7, 12, 13, 19, 22, 7]])
    similarity = es.similarity(a)
    assert round(similarity[0, 0], 3) == 0
    assert round(similarity[1, 0], 3) == 12.410
    assert round(similarity[0, 1], 3) == 0
    assert round(similarity[1, 1], 3) == 0



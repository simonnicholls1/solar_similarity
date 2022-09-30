from typing import List

import numpy as np
import numpy.typing as npt

class Analysis:

    def __init__(self):
        pass

    def top_n_similar_pairs(self, score_matrix: npt.NDArray, n: int, labels: List):
        # Need to set diagonal to very low number
        scores = np.array(score_matrix, copy=True)
        np.fill_diagonal(scores, -999)
        top_n_idx = np.argpartition(scores.ravel(), scores.size - n)[-n:]
        indices = np.column_stack(np.unravel_index(top_n_idx, scores.shape))
        return_list = indices
        if labels is not None:
            return_list = [(labels[x[0]], labels[x[1]]) for x in indices]
        return return_list

    def pair_most_like_given(self, score_matrix: npt.NDArray, n: int, labels: List):
        comparison_idx = labels.index(n)
        similarities = score_matrix[comparison_idx][0:comparison_idx]
        max_index_col = np.argmax(similarities, axis=0)
        most_common_year = labels[max_index_col]
        return most_common_year

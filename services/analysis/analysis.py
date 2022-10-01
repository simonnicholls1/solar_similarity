from typing import List

import numpy as np
import numpy.typing as npt

class Analysis:

    def __init__(self):
        pass

    def top_n_similar_pairs(self, score_matrix: npt.NDArray, n: int, labels: List, max_similar_val: bool):
        # Need to set diagonal to very low number
        scores = np.array(score_matrix, copy=True)
        if max_similar_val is not True:
            # Checking if similar scores are high better or lower better
            # If low score is better then multiply by -1 as we look for max
            scores = scores * -1
            scores[scores == 0] = -99999
        np.fill_diagonal(scores, -99999)
        top_n_idx = np.argpartition(scores.ravel(), scores.size - n)[-n:]
        indices = np.column_stack(np.unravel_index(top_n_idx, scores.shape))
        return_list = indices
        return_list = [(labels[x[0]], labels[x[1]]) for x in indices]
        return return_list

    def pair_most_like_given(self, score_matrix: npt.NDArray, n: int, labels: List, max_similar_val: bool):
        comparison_idx = labels.index(n)
        similarities = score_matrix[comparison_idx][0:comparison_idx]

        if max_similar_val is not True:
            # Checking if similar scores are high better or lower better
            # If low score is better then multiply by -1 as we look for max
            similarities = similarities * -1

        max_index_col = np.argmax(similarities, axis=0)
        most_common_year = labels[max_index_col]
        return most_common_year

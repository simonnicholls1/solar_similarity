from typing import List
import numpy as np
import numpy.typing as npt


class Analysis:

    def __init__(self):
        pass

    def top_n_similar_pairs(self, score_matrix: npt.NDArray, n: int, labels: List[str], max_similar_val: bool):
        """
        Provides the top n pairs selected from given score matrix

        param npt.NDArray score_matrix: Matrix of similarity scores
        param int n: top n pairs
        param List[str] labels: labels for columns
        param bool max_similar_val: simimlarity scores are max yes is True
        """

        if len(score_matrix) == 0:
            raise ValueError('Score matrix is empty!')

        scores = np.array(score_matrix, copy=True)
        if max_similar_val is not True:
            # Checking if similar scores are high better or lower better
            # If low score is better then multiply by -1 as we look for max
            scores = scores * -1
            scores[scores == 0] = -99999

        # Need to set diagonal to very low number
        np.fill_diagonal(scores, -99999)
        # First ravel the matrix then get the n top scores
        top_n_idx = np.argpartition(scores.ravel(), scores.size - n)[-n:]
        # Then we can put it back to get the indices of those top scores
        indices = np.column_stack(np.unravel_index(top_n_idx, scores.shape))
        # Match indices with labels
        return_list = [(labels[x[0]], labels[x[1]]) for x in indices]
        return return_list

    def pair_most_like_given(self, score_matrix: npt.NDArray, n: int, labels: List, max_similar_val: bool):
        """
        Provides the best pair for a given index, i.e given the year 2012 find me the best matched other year

        param npt.NDArray score_matrix: Matrix of similarity scores
        param int n: top n pairs
        param List[str] labels: labels for columns
        param bool max_similar_val: simimlarity scores are max yes is True
        """
        if len(score_matrix) == 0:
            raise ValueError('Score matrix is empty!')

        comparison_idx = labels.index(n)
        similarities = score_matrix[comparison_idx][0:comparison_idx]

        if max_similar_val is not True:
            # Checking if similar scores are high better or lower better
            # If low score is better then multiply by -1 as we look for max
            similarities = similarities * -1

        max_index_col = np.argmax(similarities, axis=0)
        most_common_year = labels[max_index_col]
        return most_common_year

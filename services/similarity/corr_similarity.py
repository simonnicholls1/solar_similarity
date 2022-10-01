import pandas as pd
from services.similarity.similarity import Similarity
import numpy as np


class CorrelationSimilarity(Similarity):

    def __init__(self):
        pass

    def similarity(self, data_frame: pd.DataFrame):
        variable_matrix = data_frame.to_numpy()
        distances = np.corrcoef(variable_matrix)
        distances = np.tril(distances)
        return distances


import pandas as pd
from services.similarity.similarity import  Similarity
import numpy as np

class CosineSimilarity(Similarity):

    def __init__(self):
        pass

    def similarity(self, data_frame: pd.DataFrame):
        variable_matrix = data_frame.to_numpy()
        n = len(variable_matrix)
        similarity_matrix = np.zeros((n, n))
        for i in range(0, n):
            for t in range(0, n-(n-i)+1):
                similarity = self.calculate_similarity(variable_matrix[i], variable_matrix[t])
                similarity_matrix[i, t] = similarity
        return similarity_matrix

    def calculate_similarity(self, vector_a, vector_b):
        dot_prod = self.calculate_dot_product(vector_a, vector_b)
        cross_prod = self.calculate_cross_product(vector_a, vector_b)
        return dot_prod / cross_prod

    def calculate_dot_product(self, vector_a, vector_b):
        n = len(vector_a)
        if n != len(vector_b):
            raise ValueError("Lenghts of input vectors different!")

        total_sum = 0
        for i in range(0, n):
            total_sum += vector_a[i] * vector_b[i]
        return total_sum

    def calculate_vector_length(self, vector):
        n = len(vector)
        length = 0

        for i in range(n):
            length += vector[i] ** 2
        return length ** 0.5

    def calculate_cross_product(self, vector_a, vector_b):
        n = len(vector_a)
        if n != len(vector_b):
            raise ValueError("Lenghts of input vectors different!")

        a_length = self.calculate_vector_length(vector_a)
        b_length = self.calculate_vector_length(vector_b)

        cross_product = (a_length * b_length)
        return cross_product

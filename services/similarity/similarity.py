from abc import ABC, abstractmethod


class Similarity(ABC):

    @abstractmethod
    def similarity(self, data_frame):
        pass
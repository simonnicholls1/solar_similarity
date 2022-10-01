from abc import abstractmethod, ABC


class SolarDAO(ABC):

    @abstractmethod
    def solar_data(self):
        pass

    @abstractmethod
    def get_solar_data(self, drop_non_complete_current_year=True):
        pass
import pandas as pd
import csv
import io
import requests
from config.config import params
from data.dal.solar_dao import SolarDAO

class SolarFTPDAO(SolarDAO):

    def __init__(self, env):
        self.url = params[env]['solar_url']
        self.sep = " "
        self.columns = ['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        self._solar_data = self.get_solar_data()

    @property
    def solar_data(self):
        return self._solar_data

    def get_solar_data(self, drop_non_complete_current_year=True):
        file = self.get_file_from_url()
        data = self.parse_solar_csv(file)
        df = pd.DataFrame(data=data, columns=self.columns)
        # Might want some error handling here - any casting is always risky!
        df = df.apply(pd.to_numeric, errors='coerce')
        df.set_index('year', inplace=True)
        # Check latest year - if we haven't got a full year should probably drop it
        if drop_non_complete_current_year:
            df.drop(df.index[-1], inplace=True)
        return df

    def get_file_from_url(self):
        r = requests.get(self.url)
        buff = io.StringIO(r.text)
        return buff

    def parse_solar_csv(self, solar_csv_file):
        '''Specific parser for the solar csv file
           which has a specific layout'''

        reader = csv.reader(solar_csv_file)

        # We can get the start and end years from the first row
        first_row = next(reader)[0].split()
        start_year = first_row[0]
        end_year = first_row[1]
        no_years = int(end_year) - int(start_year)
        if no_years < 0:
            raise ValueError("Start year ({0}) and end year ({1}) issue".format(start_year, end_year))

        data = []
        for i in range(0, no_years+1):
            line = next(reader)[0].split()
            # Double space between first column and second remove redundant value
            data.append(line)

        # Check final year in data is final year
        if (data[-1][0] != end_year):
            raise ValueError("Expected final year: {0} does not match data final year: {1}".format(end_year, data[1][0]))

        # Get the last row check it is -999 which signifies end of data
        data_end_row = next(reader)
        if data_end_row[0].strip() != "-999":
            raise ValueError("Did not find closing row with -999 at end of data section please check file.")

        return data



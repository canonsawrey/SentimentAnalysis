"""

Implementation for a DataSource object that uses CSV data

"""


# Libraries
from csv import reader

# Locals
from data_source import DataSource, Datum


class CSVFileDataSource(DataSource):
    # CSV file must be of format SENTENCE, SENTIMENT
    def __init__(self, file: str):
        self.data = []
        self.file = file

    def load(self):
        with open(self.file) as csv_file:
            csv_reader = reader(csv_file, delimiter=',')
            for line in csv_reader:
                self.data.append(Datum(line[0], int(line[1])))
        print(f'Data from {self.file} read into DataSource')

    def list_data(self):
        return self.data

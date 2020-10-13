"""

Implementation for a DataSource object that uses CSV data

"""


# Libraries
from csv import reader
import pandas as pd

# Locals
from training_data_source import TrainingDataSource, TrainingDatum


class CSVFileTrainingDataSource(TrainingDataSource):
    # CSV file must be of format SENTENCE, SENTIMENT
    def __init__(self, file: str):
        self.data = []
        self.file = file

    def load(self):
        with open(self.file) as csv_file:
            csv_reader = reader(csv_file, delimiter=',')
            for line in csv_reader:
                self.data.append(TrainingDatum(line[0], int(line[1])))
        print(f'Data from {self.file} loaded')

    def list_data(self):
        return self.data

    # Returns a DF of datum
    def df_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file)
        return df.rename(columns={df.columns[0]: 'sentences', df.columns[1]: 'labels'})

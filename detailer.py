import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class AdsDataDetail:
    def __init__(self):
        pass

    def read_csv(self, path):
        df = pd.read_csv(path)
        nan_percent = df.isna().mean() * 100
        print(nan_percent)

if __name__ == '__main__':
    detailer = AdsDataDetail()
    detailer.read_csv('data.csv')